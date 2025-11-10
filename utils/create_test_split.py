import sys
import pandas as pd
import random

def create_paired_split(input_csv, train_output_csv, test_output_csv, sample_size=100):
    """
    Cria um conjunto de treino selecionando N pares (H e N)
    e coloca todo o resto em um conjunto de teste.
    """
    
    try:
        df = pd.read_csv(input_csv)
        if 'id' not in df.columns:
            print("ERRO: Coluna 'id' não encontrada no CSV.")
            return

        df['id'] = df['id'].astype(str)
        df['base_id'] = df['id'].str[:-2]
        df['suffix'] = df['id'].str[-2:]

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {input_csv}")
        return
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return

    ids_h = set(df[df['suffix'] == '.H']['base_id'])
    ids_n = set(df[df['suffix'] == '.N']['base_id'])
    
    complete_pairs_base_ids = list(ids_h.intersection(ids_n))
    
    if len(complete_pairs_base_ids) < sample_size:
        print(f"ERRO: Não há pares suficientes para amostragem.")
        print(f"    Solicitado: {sample_size} pares")
        print(f"    Encontrado: {len(complete_pairs_base_ids)} pares completos")
        return

    random.seed(42)
    selected_base_ids_for_train = set(random.sample(complete_pairs_base_ids, sample_size))

    df_train = df[df['base_id'].isin(selected_base_ids_for_train)]
    
    df_test = df[~df['base_id'].isin(selected_base_ids_for_train)]

    df_train = df_train.drop(columns=['base_id', 'suffix'])
    df_test = df_test.drop(columns=['base_id', 'suffix'])


    try:
        df_train.to_csv(train_output_csv, index=False)
        print(f"Conjunto de TREINO salvo em '{train_output_csv}' ({len(df_train)} linhas, {sample_size} pares)")
        
        df_test.to_csv(test_output_csv, index=False)
        print(f"Conjunto de TESTE salvo em '{test_output_csv}' ({len(df_test)} linhas)")
        
    except Exception as e:
        print(f"ERRO ao salvar arquivos: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python create_test_split.py <arquivo_entrada.csv> <arquivo_treino.csv> <arquivo_teste.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    
    create_paired_split(input_file, train_file, test_file, sample_size=5)