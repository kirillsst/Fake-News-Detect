from rag_system.azure_client import client

def generate_response(context_text: str, user_text: str, max_tokens: int = 500):
    """
    Génère une réponse en combinant le contexte et la requête utilisateur.
    """
    prompt = f"Context: {context_text}\n\nUser question: {user_text}\nAnswer:"
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": """ ou are a professional fake news detection AI. 
                Your goal is to classify a statement as either TRUE or FAKE based on the provided context.

                Rules:
                - If the context clearly supports the statement, answer TRUE.
                - If the context contradicts the statement, answer FAKE.
                - If there is no relevant context, answer UNKNOWN.
                - Never guess or invent facts.

                Output format:
                VERDICT: TRUE or FAKE
                EXPLANATION: one short sentence explaining your reasoning.
            """},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=max_tokens
    )
    return response.choices[0].message.content
