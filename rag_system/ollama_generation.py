import ollama

MODEL_NAME = "phi3:mini"

def generate_response(clean_text: str, context_text: str) -> str:
    """
    Formule une invite stricte et appelle Ollama pour analyser le texte.
    Retourne le verdict (TRUE/FAKE) et l'explication.
    """
    prompt = f"""
You are a professional fake news detection AI. 
Your goal is to classify a statement as either TRUE or FAKE based on the provided context.

Rules:
- If the context clearly supports the statement, answer TRUE.
- If the context contradicts the statement, answer FAKE.
- If there is no relevant context, answer UNKNOWN.
- Never guess or invent facts.

Output format:
VERDICT: TRUE or FAKE
EXPLANATION: one short sentence explaining your reasoning.

User Statement:
{clean_text}

Context (related articles):
{context_text}
"""

    # Appel du modèle Ollama
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"]

    # Extraction du verdict
    verdict_line = [line for line in response_text.splitlines() if line.strip().upper().startswith("VERDICT:")]
    verdict = verdict_line[0].split(":", 1)[1].strip() if verdict_line else "UNKNOWN"

    # Extraction de l'explication
    explanation_line = [line for line in response_text.splitlines() if line.strip().upper().startswith("EXPLANATION:")]
    explanation = explanation_line[0].split(":", 1)[1].strip() if explanation_line else ""

    return f"Verdict: {verdict}\nExplanation: {explanation}"