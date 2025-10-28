# ollama_generation.py: Génération du verdict avec Ollama

import ollama

def generate_verdict(user_query: str, context_text: str, model: str = "phi3:mini") -> str:
    """
    Génère le verdict TRUE/FAKE avec justification à partir du contexte
    """
    prompt = f"""
    You are a fact-checker AI.
    Using the following context from news articles, analyze the user's text and determine if it is TRUE or FAKE.
    Provide a verdict (TRUE or FAKE) and a short explanation.

    User Text:
    {user_query}

    Context from similar articles:
    {context_text}
    """
    response = ollama.generate(model=model, prompt=prompt)
    if hasattr(response, "response"):
        return response.response.strip()
    return str(response).strip()
