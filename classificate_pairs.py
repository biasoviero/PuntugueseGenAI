import sys
import ollama
import pandas as pd
import sqlite3
import re
import random
from tqdm import tqdm

MODEL_NAME = "llama3" 

def parse_llm_response(response_text):
    """
    Extrai tuplas (texto, rótulo) lidando corretamente com frases que contêm vírgulas.
    Usa a última vírgula da tupla como separador.
    """
    raw_matches = re.findall(r"\((.*?)\)", response_text, re.DOTALL)
    
    predicted_pun = None
    predicted_non_pun = None
    
    for match in raw_matches:
        
        last_comma_index = match.rfind(',')
        
        if last_comma_index == -1:
            continue
            
        text_part = match[:last_comma_index].strip()
        label_part = match[last_comma_index+1:].strip()
        
        clean_text = text_part.strip().strip("'").strip('"')
        clean_label = label_part.strip().strip("'").strip('"').lower()
        
        if 'não' in clean_label or 'nao' in clean_label or 'non' in clean_label:
            predicted_non_pun = clean_text
        else:
            predicted_pun = clean_text
            
    return predicted_pun, predicted_non_pun

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair_id TEXT,
        pun_phrase_gold TEXT,
        non_pun_phrase_gold TEXT,
        
        model_response_raw TEXT,
        predicted_pun_phrase TEXT,
        
        is_correct INTEGER, -- 1 se acertou, 0 se errou
        error_flag INTEGER DEFAULT 0 -- 1 se houve erro de parse
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pair_id ON results_pairs (pair_id)")
    conn.commit()
    conn.close()

def get_processed_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT pair_id FROM results_pairs")
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()

def process_pairs_csv(csv_path, prompt_template_path, db_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return

    pairs_dict = {}
    for _, row in df.iterrows():
        full_id = str(row.get('id', ''))
        text = str(row.get('text', ''))
        
        if '.' in full_id:
            base_id, suffix = full_id.rsplit('.', 1)
            if base_id not in pairs_dict:
                pairs_dict[base_id] = {}
            
            if suffix == 'H':
                pairs_dict[base_id]['pun'] = text
            elif suffix == 'N':
                pairs_dict[base_id]['non'] = text

    setup_database(db_path)
    processed_ids = get_processed_ids(db_path)
    
    pairs_to_process = []
    for pid, data in pairs_dict.items():
        if 'pun' in data and 'non' in data:
            if pid not in processed_ids:
                pairs_to_process.append((pid, data['pun'], data['non']))
    
    print(f"Total de pares encontrados: {len(pairs_dict)}")
    print(f"Pares a processar: {len(pairs_to_process)}")
    
    if not pairs_to_process:
        return

    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_instruction = f.read()
    except:
        print("Erro ao ler arquivo de prompt.")
        return

    system_instruction = """
    Você é um classificador linguístico.
    Regra: Receba duas frases. Uma é Trocadilho, a outra Não.
    Saída OBRIGATÓRIA:
    (Texto da frase, Trocadilho)
    (Texto da frase, Não trocadilho)
    Não explique. Apenas as tuplas.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for pair_id, gold_pun, gold_non in tqdm(pairs_to_process):
        
        phrases_list = [gold_pun, gold_non]
        random.shuffle(phrases_list)
        
        final_prompt = f"{prompt_instruction}\n\nFrases:\n1. {phrases_list[0]}\n2. {phrases_list[1]}"

        ollama_options = {
                "temperature": 0,
                "seed": 42
            }
        
        system_prompt = ''''
        Siga estritamente este formato de resposta, sem adicionar texto extra:
        (Texto da frase, Classificação)
        (Texto da frase, Classificação)
        '''

        try:
            response = ollama.generate(
                model="llama3",
                system=system_prompt,
                prompt=final_prompt,
                options=ollama_options,
                stream=False
            )
            
            raw_response = response['response'].strip()
            
            pred_pun, pred_non = parse_llm_response(raw_response)
            
            is_correct = 0
            error_flag = 0
            
            if pred_pun:
                def normalize_text(t):
                    return t.lower().strip().strip("'").strip('"').strip('.').strip()

                p_clean = normalize_text(pred_pun)
                g_clean = normalize_text(gold_pun)
                
                if p_clean == g_clean or (len(p_clean) > 4 and p_clean in g_clean):
                    is_correct = 1
                else:

                    pass
            else:
                error_flag = 1
            
            cursor.execute("""
                INSERT INTO results_pairs 
                (pair_id, pun_phrase_gold, non_pun_phrase_gold, model_response_raw, predicted_pun_phrase, is_correct, error_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pair_id, gold_pun, gold_non, raw_response, pred_pun, is_correct, error_flag))
            
            conn.commit()

        except Exception as e:
            print(f"Erro no par {pair_id}: {e}")
            continue

    conn.close()
    print("Processamento finalizado.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python classificate_pairs.py <dataset.csv> <prompt.txt> <output.db>")
    else:
        process_pairs_csv(sys.argv[1], sys.argv[2], sys.argv[3])