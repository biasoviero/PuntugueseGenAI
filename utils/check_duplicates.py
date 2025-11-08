import sys
import pandas as pd

def remove_text_duplicates(csv_file_path, output_csv_path):
    """
    Loads a CSV and prints any rows that have a duplicated 'text' field.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"ERROR: The file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    df_deduplicated = df.drop_duplicates(subset=['text'], keep='first')

    num_removed = len(df) - len(df_deduplicated)
    if num_removed == 0:
        print("No duplicate texts found.")
    else:
        print(f"Found and removed {num_removed} duplicate text row(s).")

    try:
        df_deduplicated.to_csv(output_csv_path, index=False)
        print(f"Clean data saved to '{output_csv_path}' ({len(df_deduplicated)} rows).")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_duplicates.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    remove_text_duplicates(input_path, output_path)