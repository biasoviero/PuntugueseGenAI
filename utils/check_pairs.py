import sys
import pandas as pd

def validate_pairs(csv_file_path):
    """
    Verifica um arquivo CSV para garantir que cada ID 'base'
    tenha um sufixo '.H' e um sufixo '.N' correspondentes.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {csv_file_path}")
        return
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return

    if 'id' not in df.columns:
        print("ERRO: Coluna 'id' não encontrada no CSV.")
        return

    all_ids = set(df['id'])


    orphaned_h = []
    orphaned_n = []
    unrecognized = [] 

    for current_id in all_ids:
        
        if isinstance(current_id, str) and current_id.endswith('.H'):
            base_id = current_id[:-2]
            pair_id = base_id + '.N'
            
            if pair_id not in all_ids:
                orphaned_h.append(current_id)
                
        elif isinstance(current_id, str) and current_id.endswith('.N'):
            base_id = current_id[:-2]
            pair_id = base_id + '.H'
            
            if pair_id not in all_ids:
                orphaned_n.append(current_id)
        
        elif not isinstance(current_id, str):
             unrecognized.append(f"ID não-string: {current_id}")
        else:
            unrecognized.append(f"ID com formato inválido: {current_id}")

    if not orphaned_h and not orphaned_n and not unrecognized:
        print(f"Sucesso! Todos os {len(all_ids)} IDs estão pareados corretamente.")
        print(f"(Total de {len(all_ids) // 2} pares encontrados)")
    else:
        print("🚨 Verificação falhou. Encontrados os seguintes problemas:")
        
        if orphaned_h:
            print(f"\n{len(orphaned_h)} IDs '.H' sem um par '.N':")
            for item in orphaned_h:
                print(f"  - {item} (par faltante: {item[:-2]}.N)")
                
        if orphaned_n:
            print(f"\n{len(orphaned_n)} IDs '.N' sem um par '.H':")
            for item in orphaned_n:
                print(f"  - {item} (par faltante: {item[:-2]}.H)")
                
        if unrecognized:
            print(f"\n{len(unrecognized)} IDs com formato não reconhecido:")
            for item in unrecognized:
                print(f"  - {item}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python check_pairs.py <caminho_para_seu_csv.csv>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    validate_pairs(file_path)