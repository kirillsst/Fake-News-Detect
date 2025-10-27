# ollama_generation.py Génération de la réponse finale via Ollama

import ollama

MODEL_NAME = "phi3:mini"

def generate_response(clean_text: str, context_text: str) -> str:
    """
    Formule une invite et appelle Ollama pour analyser le texte.
    """
    prompt = f"""
You are a fact-checker AI. Using the following context from news articles, 
analyze the user's text and determine whether it is TRUE or FAKE.
Provide a verdict (TRUE or FAKE) and a short explanation.

User Text:
{clean_text}

Context from similar articles:
{context_text}
"""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
