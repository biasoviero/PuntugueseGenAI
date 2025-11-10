import sys
import ollama

def run_prompt_from_file(prompt_filepath, output_filepath):
    """
    Lê um prompt de um arquivo e executa o ollama.generate com ele.
    """
    try:
        with open(prompt_filepath, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
            
    except FileNotFoundError:
        print(f"ERRO: O arquivo de prompt '{prompt_filepath}' não foi encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao ler o arquivo: {e}")
        sys.exit(1)

    options = {
        "temperature": 0,
        "seed": 42
    }

    print(f"--- Prompt carregado de '{prompt_filepath}' ---")
    print("--- Gerando resposta... ---")

    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt_text,
            options=options,
            stream=False
        )

        response_text = response["response"]
        print("\n--- Resposta do Modelo ---")
        print(response_text)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(response_text)
            print(f"Resposta salva com sucesso em '{output_filepath}'.")
            
        except Exception as e:
            print(f"ERRO: Ocorreu um erro ao salvar o arquivo de resposta: {e}")

    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao chamar a API do Ollama: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python run_prompt_file.py <caminho_para_seu_prompt.txt> <arquivo_resposta.txt>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    run_prompt_from_file(input_path, output_path)