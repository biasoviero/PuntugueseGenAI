import sys
import ollama
import pandas as pd
import sqlite3
import re
import random
from tqdm import tqdm

def parse_tuple_response(response_str):
    """
    Extracts tuples in (phrase, label) format from model response.
    Processes responses with exactly 2 tuples (one for each phrase in the pair).
    Returns phrases regardless of the order the model classified them.
    
    Returns: (pun_phrase, non_pun_phrase)
    """
    
    response = response_str.strip()
    
    tuples = []
    depth = 0
    current = ""
    
    for char in response:
        if char == '(':
            depth += 1
            if depth == 1:
                current = "("
            else:
                current += char
        elif char == ')':
            if depth == 1:
                current += ")"
                tuples.append(current)
                current = ""
            else:
                current += char
            depth -= 1
        elif depth > 0:
            current += char
    
    if len(tuples) < 2:
        raise ValueError(f"Esperado pelo menos 2 tuplas; encontrado {len(tuples)}")
    
    phrases_with_labels = []
    
    for tuple_str in tuples[:2]: 
        content = tuple_str[1:-1].strip()
        last_comma = content.rfind(',')
        if last_comma == -1:
            raise ValueError(f"Nenhuma vírgula encontrada em: {tuple_str}")
        
        phrase_part = content[:last_comma].strip()
        label_part = content[last_comma + 1:].strip()
        
        phrase = phrase_part.strip("'\"")
        label = label_part.strip("'\"")
        
        label_lower = label.lower().strip()
        if 'não' in label_lower or 'nao' in label_lower:
            label_norm = 'Não trocadilho'
        else:
            label_norm = 'Trocadilho'
        
        phrases_with_labels.append((phrase, label_norm))
    
    pun_phrase = None
    non_pun_phrase = None
    
    for phrase, label in phrases_with_labels:
        if label == 'Trocadilho':
            pun_phrase = phrase
        elif label == 'Não trocadilho':
            non_pun_phrase = phrase
    
    if pun_phrase is None or non_pun_phrase is None:
        if pun_phrase is not None and non_pun_phrase is None:
            non_pun_phrase = phrases_with_labels[1][0]
        elif non_pun_phrase is not None and pun_phrase is None:
            pun_phrase = phrases_with_labels[0][0]
        else:
            raise ValueError(f"Não foi possível extrair corretamente ambas as frases classificadas. Trocadilho: {pun_phrase is not None}, Não trocadilho: {non_pun_phrase is not None}")
    
    return pun_phrase, non_pun_phrase

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair_id TEXT UNIQUE,
        pun_phrase TEXT,
        non_pun_phrase TEXT,
        correct_label TEXT, 
        model_input_prompt TEXT,
        model_response_raw TEXT,
        extracted_pun_phrase TEXT,
        extracted_non_pun_phrase TEXT,
        extracted_label TEXT,
        extracted_chosen_index INTEGER,
        extracted_chosen_phrase TEXT
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pair_id ON results_pairs (pair_id)")
    conn.commit()
    conn.close()

def process_pairs_csv(csv_path, prompt_template_path, db_path):
    """
    Processes pairs of phrases from a CSV where each pair has two consecutive lines
    (one with .H for Pun and another with .N for Non-pun).
    Uses Llama 3 to classify and saves to SQLite.
    """
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo CSV '{csv_path}' não encontrado.")
        return
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return

    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"ERRO: Arquivo de prompt '{prompt_template_path}' não encontrado.")
        return

    setup_database(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT pair_id FROM results_pairs")
    processed_pairs = {row[0] for row in cursor.fetchall()}
    print(f"Encontrados {len(processed_pairs)} pares já processados no banco de dados.")

    pairs_dict = {}
    
    for _, row in df.iterrows():
        row_id = str(row.get('id', 'N/A'))
        text = str(row.get('text', 'N/A'))
        label = str(row.get('label', 'N/A'))
        
        if row_id.endswith('.H'):
            base_id = row_id[:-2]
            if base_id not in pairs_dict:
                pairs_dict[base_id] = {}
            pairs_dict[base_id]['trocadilho'] = text
        elif row_id.endswith('.N'):
            base_id = row_id[:-2]
            if base_id not in pairs_dict:
                pairs_dict[base_id] = {}
            pairs_dict[base_id]['nao_trocadilho'] = text
    
    pairs_to_process = []
    for base_id, pair_data in sorted(pairs_dict.items()):
        if 'trocadilho' in pair_data and 'nao_trocadilho' in pair_data and base_id not in processed_pairs:
            pairs_to_process.append((base_id, pair_data['trocadilho'], pair_data['nao_trocadilho']))
    
    if len(pairs_to_process) == 0:
        print("Nenhum par novo para processar. Encerrando.")
        conn.close()
        return
        
    print(f"--- Processando {len(pairs_to_process)} NOVOS pares de {len(pairs_dict)} totais ---")
    
    ollama_options = {
        "temperature": 0,
        "seed": 42
    }
    
    system_prompt = "Responda APENAS com a tupla solicitada. Não inclua nenhum outro texto."

    for pair_id, pun_phrase, non_pun_phrase in tqdm(pairs_to_process, total=len(pairs_to_process)):
        
        phrases = [pun_phrase, non_pun_phrase]
        random.shuffle(phrases)
        
        pair_text = f"Frase 1: {phrases[0]}\nFrase 2: {phrases[1]}"
        
        try:
            final_prompt = f"{prompt_template}\n{pair_text}"
        except KeyError:
            print(f"\nERRO: Seu prompt.txt NÃO contém a tag adequada. Verifique o arquivo.")
            break

            
        model_response_raw = ""
        extracted_pun_phrase = "PARSE_ERROR"
        extracted_non_pun_phrase = "PARSE_ERROR"

        try:
            response = ollama.generate(
                model="llama3",
                system=system_prompt,
                prompt=final_prompt,
                options=ollama_options,
                stream=False
            )
            
            model_response_raw = response['response'].strip()
            
            try:
                extracted_pun_phrase, extracted_non_pun_phrase = parse_tuple_response(model_response_raw)
            except (ValueError, TypeError) as parse_error:
                print(f"\nAVISO: Erro ao processar resposta: '{model_response_raw}'. Erro: {parse_error}")

        except Exception as e:
            print(f"\nERRO Inesperado: {e}")
            break

        cursor.execute(
            """INSERT INTO results_pairs 
               (pair_id, pun_phrase, non_pun_phrase, 
                model_input_prompt, model_response_raw, extracted_pun_phrase, 
                extracted_non_pun_phrase) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pair_id, pun_phrase, non_pun_phrase, 
             final_prompt, model_response_raw, extracted_pun_phrase,
             extracted_non_pun_phrase)
        )
        
        conn.commit()
    conn.close()
    print(f"\n Processamento concluído. Resultados salvos em '{db_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python classificate_pairs.py <arquivo.csv> <prompt.txt> <banco.db>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    prompt_file = sys.argv[2]
    db_file = sys.argv[3]
    
    process_pairs_csv(csv_file, prompt_file, db_file)
