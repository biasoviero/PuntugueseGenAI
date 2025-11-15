import sys
import pandas as pd
import csv

def convert_csv_to_txt_tuples(csv_path, txt_path, label_filter=None):
    """
    Converte colunas ('text', 'label', 'pun_sign', 'alternative_sign')
    de um arquivo CSV para um arquivo TXT, onde cada linha é um tuple Python.

    Args:
        csv_path (str): O caminho para o arquivo CSV de entrada.
        txt_path (str): O caminho para o arquivo TXT de saída.
        label_filter (str, optional): Um label para filtrar os textos (ex: 'Trocadilho').
                                      Se None, todas as linhas são extraídas.
    """
    try:
        df = pd.read_csv(csv_path)

        columns_to_export = ['text', 'label', 'pun_sign', 'alternative_sign']

        # 1. Verificar se todas as colunas necessárias existem
        missing_cols = [col for col in columns_to_export if col not in df.columns]
        if missing_cols:
            print(f"ERRO: Colunas não encontradas no CSV: {', '.join(missing_cols)}")
            print("Verifique se o seu CSV contém 'text', 'label', 'pun_sign' e 'alternative_sign'.")
            return

        df_to_export = df

        # 2. Aplicar filtro, se fornecido
        if label_filter:
            # A coluna 'label' já foi verificada acima
            # Usamos .copy() para evitar avisos de SettingWithCopyWarning
            df_to_export = df[df['label'] == label_filter].copy()
            print(f"Filtrando por label: '{label_filter}'...")
            if df_to_export.empty:
                print(f"Aviso: Nenhum dado encontrado com o label '{label_filter}'.")
        else:
            df_to_export = df.copy()
            print("Exportando todas as linhas (sem filtro)...")

        # 3. Preparar dados
        # Selecionar apenas as colunas desejadas
        df_to_export = df_to_export[columns_to_export]
        # Substituir valores NaN (como em pun_sign) por strings vazias
        # Isso garante que o output seja ('...','...', '', '') em vez de ('...','...', nan, nan)
        df_to_export.fillna('', inplace=True)

        # 4. Escrever no arquivo TXT
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Iterar sobre cada linha do DataFrame
            for row in df_to_export.itertuples(index=False):
                # row é agora um tuple com os 4 valores (ex: ('Texto...', 'Trocadilho', 'pun', 'alt'))
                
                # Usar repr() em cada valor para formatá-lo como uma string Python
                # Isso cuida automaticamente de aspas, escapes, etc.
                # Ex: "O'Brien" vira "'O\'Brien'"
                # Ex: 123 vira "123"
                # Ex: '' vira "''"
                tuple_values = [repr(value) for value in row]
                
                # Juntar os valores formatados com vírgula e espaço
                # e envolver com parênteses
                tuple_string = f"({', '.join(tuple_values)})"
                
                # Escrever a linha final no arquivo
                f.write(tuple_string + '\n')
        
        print(f"Arquivo TXT '{txt_path}' criado com sucesso com {len(df_to_export)} linhas.")

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{csv_path}' não foi encontrado.")
    except pd.errors.EmptyDataError:
        print(f"ERRO: O arquivo CSV '{csv_path}' está vazio.")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

if __name__ == "__main__":

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nUso via linha de comando:")
        print("  Modo 1 (Extrair tudo):   python seu_script.py <arquivo_csv_entrada> <arquivo_txt_saida>")
        print("  Modo 2 (Filtrar):        python seu_script.py <arquivo_csv_entrada> <arquivo_txt_saida> <label_para_filtrar>")
        print("\nExemplo (Modo 2):")
        print("  python seu_script.py dados.csv textos_filtrados.txt Trocadilho")
        sys.exit(1)

    csv_input_path = sys.argv[1]
    txt_output_path = sys.argv[2]

    label_to_filter = None
    if len(sys.argv) == 4:
        label_to_filter = sys.argv[3]
    
    # Renomeei a função para ficar mais claro
    convert_csv_to_txt_tuples(csv_input_path, txt_output_path, label_to_filter)