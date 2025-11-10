import sys
import pandas as pd

def convert_csv_to_txt(input_csv, output_txt):
    """
    Lê um arquivo CSV e salva as colunas 'text' e 'label'
    em um arquivo TXT no formato (texto, label).
    """
    try:
        df = pd.read_csv(input_csv)

        if 'text' not in df.columns or 'label' not in df.columns:
            print("ERRO: O CSV deve conter as colunas 'text' e 'label'.")
            return

        with open(output_txt, 'w', encoding='utf-8') as f:
            
            for index, row in df.iterrows():
                
                text_data = str(row['text']) 
                label_data = str(row['label']) 

                new_line = f"({repr(text_data)}, {repr(label_data)})\n"
                
                f.write(new_line)
        
        print(f"{len(df)} linhas salvas com sucesso em '{output_txt}'.")

    except FileNotFoundError:
        print(f"ERRO: Arquivo de entrada '{input_csv}' não encontrado.")
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python format_to_txt.py <arquivo_entrada.csv> <arquivo_saida.txt>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_csv_to_txt(input_path, output_path)