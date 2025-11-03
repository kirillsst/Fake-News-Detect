from rag_system.azure_client import client

def generate_response(context_text: str, user_text: str, max_tokens: int = 500):
    """
    Génère une réponse en combinant le contexte et la requête utilisateur.
    """
    prompt = f"Context: {context_text}\n\nUser question: {user_text}\nAnswer:"
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=max_tokens
    )
    return response.choices[0].message.content
