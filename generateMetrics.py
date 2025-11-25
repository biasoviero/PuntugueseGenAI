import sqlite3
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import os
import argparse

def clean_label(label):
    """
    Normalizes the label strings to handle inconsistencies.
    Input formats handled: 'Trocadilho', 'Não trocadilho', "Trocadilho'", etc.
    Returns: 1 for Trocadilho (Positive), 0 for Não Trocadilho (Negative), -1 for Unknown
    """
    if pd.isna(label) or label == "":
        return -1
    
    txt = str(label).lower().strip()
    
    txt = txt.replace("'", "").replace('"', "")
    
    if "não" in txt or "nao" in txt:
        return 0 # Class: Não Trocadilho
    elif "trocadilho" in txt:
        return 1 # Class: Trocadilho
    else:
        return -1 # Unrecognized label

def analyze_database(db_path, output_csv, debug_csv=None):
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("Error: No tables found in the database.")
            return

        table_name = tables[0][0] 
        print(f"Reading from table: '{table_name}'...")

        query = f"SELECT id, original_text, correct_label, extracted_label FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("Table is empty.")
            return
        
        df['y_true'] = df['correct_label'].apply(clean_label)
        df['y_pred'] = df['extracted_label'].apply(clean_label)

        if debug_csv:
            df.to_csv(debug_csv, index=False)
            print(f"Intermediate debug data saved to: {debug_csv}")

        unknown_rows = df[(df['y_true'] == -1) | (df['y_pred'] == -1)]
        if not unknown_rows.empty:
            print(f"Warning: {len(unknown_rows)} rows contained unrecognized labels and were excluded from calculation.")
            print("Sample of excluded labels:", unknown_rows['extracted_label'].unique())
            
        valid_df = df[(df['y_true'] != -1) & (df['y_pred'] != -1)]

        if valid_df.empty:
            print("Error: No valid data remains after cleaning.")
            return

        y_true = valid_df['y_true']
        y_pred = valid_df['y_pred']

        f1_scores = f1_score(y_true, y_pred, labels=[0, 1], average=None)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=[0, 1], average=None)
        recall = recall_score(y_true, y_pred, labels=[0, 1], average=None)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        results = {
            'Metric': [
                'F1 Score (Trocadilho)', 
                'F1 Score (Não Trocadilho)',
                'Accuracy', 
                'Precision (Trocadilho)', 
                'Recall (Trocadilho)',
                'Precision (Não Trocadilho)', 
                'Recall (Não Trocadilho)',
                'True Positives (Trocadilho)', 
                'True Negatives (Não Trocadilho)', 
                'False Positives', 
                'False Negatives', 
                'Total Samples'
            ],
            'Value': [
                f1_scores[1], 
                f1_scores[0],
                accuracy, 
                precision[1], 
                recall[1], 
                precision[0], 
                recall[0],
                tp, tn, fp, fn, len(valid_df)
            ]
        }
        
        results_df = pd.DataFrame(results)

        results_df.to_csv(output_csv, index=False)
        
        print(f"Results saved to: {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate confusion matrix and F1 score from a SQLite database.")
    
    parser.add_argument("db_file", help="Path to the input SQLite .db file")
    parser.add_argument("output_csv", help="Path for the output CSV file")
    parser.add_argument("--debug", help="Path to save the intermediate CSV for debugging (optional)", default=None)
    
    args = parser.parse_args()
    
    analyze_database(args.db_file, args.output_csv, args.debug)