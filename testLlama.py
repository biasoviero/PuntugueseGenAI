import ollama

options = {
    "temperature": 0,
    "seed": 42,
    "stop": ["\n"]  # Stop after the first line (the classification)
}

prompt = """
Você receberá uma frase e deve rotulá-la como "Trocadilho" ou "Não trocadilho".

Definição de Trocadilho:
Frases que contêm uma ambiguidade semântica intencional, usada para provocar humor. Geralmente há uma palavra ou expressão que pode ter dois significados, e o contexto força uma interpretação engraçada.

Definição de Não trocadilho:
Frases que não contém trocadilho nem jogo de palavras responsável por gerar humor.

Diretrizes de classificação:
1. Identifique se há alguma palavra ou expressão com mais de um sentido possível.
2. Verifique se o contexto cria um contraste ou jogo entre esses sentidos.
3. Se o contraste gerar humor ou surpresa, classifique como Trocadilho.
4. Caso contrário, classifique como Não trocadilho.

Você deve responder no formato (frase, rótulo classificado) para a frase
"""

phrase = "O que é que sabe três vezes pior do que menta? Cravinhos."

response = ollama.generate(
    model="llama3",
    system="Você é um classificador de frases em português",
    prompt=f"{prompt} {phrase}",
    options=options,
    stream=False
)
print(response["response"])