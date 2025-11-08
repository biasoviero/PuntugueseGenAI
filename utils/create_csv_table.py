import sys
import json
import pandas as pd

def convert_json_to_csv(json_file_path, output_csv_path):
    """
    Loads a JSON file with a specific structure and converts it to a CSV.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame.from_dict(data, orient='index')

        df = df.reset_index().rename(columns={'index': 'id'})

        label_map = {
            1: "Trocadilho",
            0: "Não trocadilho"
        }

        df['label'] = df['label'].map(label_map)
        
        df.to_csv(output_csv_path, index=False)
        
        print(f"Successfully converted '{json_file_path}' to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"ERROR: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode the JSON. Check the file for syntax errors.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_json_to_csv.py <input_json_path> <output_csv_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_json_to_csv(input_path, output_path)