import sys
import pandas as pd

def get_pun_signs(file_puns_path, file_texts_path, output_path):
    """
    Filtra textos com o label 'Trocadilho' do file_texts_path e procura por 
    cada 'pun sign' do file_puns_path dentro desses textos.
    Salva os textos filtrados com uma nova coluna 'found_pun_signs' no output_path.

    Args:
        file_puns_path (str): O caminho para o CSV com os pun signs.
        file_texts_path (str): O caminho para o CSV com os textos e labels.
        output_path (str): O caminho para o CSV de saída.
    """
    try:
        df_puns = pd.read_csv(file_puns_path)
        df_texts = pd.read_csv(file_texts_path)

        if 'pun sign' not in df_puns.columns:
            print(f"ERRO: Coluna 'pun sign' não encontrada em {file_puns_path}.")
            return
        
        required_text_cols = ['text', 'label']
        if not all(col in df_texts.columns for col in required_text_cols):
            print(f"ERRO: Colunas 'text' e/ou 'label' não encontradas em {file_texts_path}.")
            return

        pun_signs_list = [str(pun).lower() for pun in df_puns['pun sign'].dropna().unique()]

        df_trocadilhos = df_texts[df_texts['label'] == 'Trocadilho'].copy()

        if df_trocadilhos.empty:
            print("Nenhum texto encontrado com o label 'Trocadilho'.")
            df_trocadilhos['found_pun_signs'] = pd.Series(dtype='object')
            df_trocadilhos.to_csv(output_path, index=False)
            print(f"Arquivo de saída vazio '{output_path}' criado.")
            return

        found_puns_col = []
        
        for text in df_trocadilhos['text']:
            found_puns_for_this_text = []
            text_lower = str(text).lower()
            
            for pun_sign in pun_signs_list:
                if pun_sign in text_lower:
                    found_puns_for_this_text.append(pun_sign)
            
            found_puns_col.append(found_puns_for_this_text)

        df_trocadilhos['found_pun_signs'] = found_puns_col
        df_trocadilhos.to_csv(output_path, index=False)
        
        print(f"Textos processados com sucesso e resultados salvos em '{output_path}'")

    except FileNotFoundError as e:
        print(f"ERRO: O arquivo '{e.filename}' não foi encontrado.")
    except pd.errors.EmptyDataError as e:
        print(f"ERRO: Um dos arquivos está vazio. {e}")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python get_pun_signs.py <file_puns_path> <file_texts_path> <output_csv_path>")
        sys.exit(1)

    file_a_path = sys.argv[1]
    file_b_path = sys.argv[2]
    output_path = sys.argv[3]
    
    get_pun_signs(file_a_path, file_b_path, output_path)