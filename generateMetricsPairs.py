import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import os
import argparse

def analyze_database(db_path, output_csv, debug_csv=None):
    if not os.path.exists(db_path):
        print(f"Erro: Arquivo de banco de dados '{db_path}' não encontrado.")
        return

    try:
        conn = sqlite3.connect(db_path)
        
        # 1. Ler apenas as colunas necessárias da tabela results_pairs
        # Filtramos onde error_flag = 0 (ignoramos erros de parse/API)
        query = "SELECT id, pair_id, is_correct FROM results_pairs"
        df_pairs = pd.read_sql_query(query, conn)
        conn.close()

        if df_pairs.empty:
            print("Aviso: A tabela está vazia ou todos os registros contêm erros (error_flag=1).")
            return
        
        print(f"Lidos {len(df_pairs)} pares válidos do banco de dados.")

        # 2. Expandir os Pares para Instâncias Individuais (Unfolding)
        # O modelo Llama recebeu 2 frases. Vamos calcular a métrica para ambas.
        
        # Lista A: As frases que ERAM Trocadilhos (Gold = 1)
        # Se is_correct=1, pred=1. Se is_correct=0, pred=0.
        y_true_pun = np.ones(len(df_pairs), dtype=int) # Tudo 1
        y_pred_pun = df_pairs['is_correct'].values     # Copia o is_correct
        
        # Lista B: As frases que ERAM Não Trocadilhos (Gold = 0)
        # Se is_correct=1, pred=0. Se is_correct=0, pred=1 (Inverteu).
        y_true_non = np.zeros(len(df_pairs), dtype=int) # Tudo 0
        y_pred_non = 1 - df_pairs['is_correct'].values  # Inverte o is_correct
        
        # Concatenar para criar o vetor final de avaliação
        y_true = np.concatenate([y_true_pun, y_true_non])
        y_pred = np.concatenate([y_pred_pun, y_pred_non])

        # 3. (Opcional) Salvar debug
        if debug_csv:
            df_debug = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            df_debug.to_csv(debug_csv, index=False)
            print(f"Dados intermediários de debug salvos em: {debug_csv}")

        # 4. Calcular Métricas
        # Labels: 1 = Trocadilho (Positivo), 0 = Não Trocadilho (Negativo)
        f1_scores = f1_score(y_true, y_pred, labels=[0, 1], average=None)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=[0, 1], average=None)
        recall = recall_score(y_true, y_pred, labels=[0, 1], average=None)

        # Matriz de Confusão
        # tn = True Negative (Não trocadilho classificado como Não trocadilho)
        # fp = False Positive (Não trocadilho classificado como Trocadilho)
        # fn = False Negative (Trocadilho classificado como Não trocadilho)
        # tp = True Positive (Trocadilho classificado como Trocadilho)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # 5. Montar Resultado
        results = {
            'Metric': [
                'F1 Score (Classe 1: Trocadilho)', 
                'F1 Score (Classe 0: Não Trocadilho)',
                'Accuracy (Global)', 
                'Precision (Trocadilho)', 
                'Recall (Trocadilho)',
                'Precision (Não Trocadilho)', 
                'Recall (Não Trocadilho)',
                'True Positives (TP)', 
                'True Negatives (TN)', 
                'False Positives (FP)', 
                'False Negatives (FN)', 
                'Total Instances (Frases)',
                'Total Pairs (Pares)'
            ],
            'Value': [
                f1_scores[1], 
                f1_scores[0],
                accuracy, 
                precision[1], 
                recall[1], 
                precision[0], 
                recall[0],
                tp, tn, fp, fn, 
                len(y_true),
                len(df_pairs)
            ]
        }
        
        results_df = pd.DataFrame(results)

        # Salvar
        results_df.to_csv(output_csv, index=False)
        
        # Exibir prévia no console
        print("-" * 30)
        print(results_df)
        print("-" * 30)
        print(f"Métricas salvas com sucesso em: {output_csv}")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcular métricas a partir do banco SQLite results_pairs.")
    
    parser.add_argument("db_file", help="Caminho para o arquivo .db")
    parser.add_argument("output_csv", help="Caminho para salvar o CSV de métricas")
    parser.add_argument("--debug", help="Caminho para salvar CSV de debug (opcional)", default=None)
    
    args = parser.parse_args()
    
    analyze_database(args.db_file, args.output_csv, args.debug)