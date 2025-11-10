import sys
import ollama
import pandas as pd
import sqlite3
import re
from tqdm import tqdm

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_text TEXT UNIQUE, 
        correct_label TEXT, 
        model_input_prompt TEXT,
        model_response_raw TEXT,
        extracted_text TEXT,
        extracted_label TEXT
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_original_text ON results (original_text)")
    conn.commit()
    conn.close()

def process_csv(csv_path, prompt_template_path, db_path):
    """
    Processa cada linha de um CSV com o Llama 3 e salva no SQLite.
    Agora, salva a cada linha e pula linhas já processadas.
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
    
    cursor.execute("SELECT original_text FROM results")
    processed_texts = {row[0] for row in cursor.fetchall()}
    print(f"Encontrados {len(processed_texts)} resultados já processados no banco de dados.")

    df_to_process = df[~df['text'].isin(processed_texts)]
    if len(df_to_process) == 0:
        print("Nenhuma linha nova para processar. Encerrando.")
        conn.close()
        return
        
    print(f"--- Processando {len(df_to_process)} NOVAS linhas de {len(df)} totais ---")
    
    ollama_options = {
        "temperature": 0,
        "seed": 42
    }
    
    system_prompt = "Responda APENAS com a tupla solicitada. Não inclua nenhum outro texto."

    for _, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0]):
        
        original_text = str(row.get('text', 'N/A'))
        correct_label = str(row.get('label', 'N/A'))

        try:
            final_prompt = f"{prompt_template}\n{original_text}"
        except KeyError:
            print(f"\nERRO: Seu prompt.txt NÃO contém a tag {{text}}. Verifique o arquivo.")
            break

            
        model_response_raw = ""
        extracted_text = "PARSE_ERROR"
        extracted_label = "PARSE_ERROR"

        try:
            response = ollama.generate(
                model="llama3",
                system=system_prompt,
                prompt=final_prompt,
                options=ollama_options,
                stream=False
            )
            
            model_response_raw = response['response'].strip()
            
            match = re.search(r'\((["\'])(.*?)\1,\s*(.*?)\s*[\)"\']*\)$', model_response_raw)
            
            if match:
                extracted_text = match.group(2)
                extracted_label = match.group(3).strip().strip('\'"') 
            else:
                raise ValueError("Regex não conseguiu encontrar o padrão (texto, label)")

        except (SyntaxError, ValueError, TypeError) as e:
            print(f"\nAVISO: Erro ao processar a resposta: '{model_response_raw}'. Erro: {e}")
            
        except Exception as e:
            print(f"\nERRO Inesperado: {e}")
            break # Sai do loop em caso de erro grave

        cursor.execute(
            "INSERT INTO results (original_text, correct_label, model_input_prompt, model_response_raw, extracted_text, extracted_label) VALUES (?, ?, ?, ?, ?, ?)",
            (original_text, correct_label, final_prompt, model_response_raw, extracted_text, extracted_label)
        )
        
        conn.commit()
    conn.close()
    print(f"\n Processamento concluído. Resultados salvos em '{db_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python classificate_phrases.py <arquivo.csv> <prompt.txt> <banco.db>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    prompt_file = sys.argv[2]
    db_file = sys.argv[3]
    
    process_csv(csv_file, prompt_file, db_file)