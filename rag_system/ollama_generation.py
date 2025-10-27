# ollama_generation.py
import ollama

MODEL_NAME = "phi3:mini"

def generate_response(clean_text: str, context_text: str) -> str:
    """
    Formule une invite stricte et appelle Ollama pour analyser le texte.
    Retourne le verdict (TRUE/FAKE) et l'explication.
    """
    prompt = f"""
You are a fact-checker AI. Using the context below, determine if the user's statement is TRUE or FAKE.
Respond strictly in the following format:

VERDICT: TRUE or FAKE
EXPLANATION: <short explanation>

User Text:
{clean_text}

Context from similar articles:
{context_text}
"""
    # Appel du modèle Ollama
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"]

    # Extraction du verdict
    verdict_line = [line for line in response_text.splitlines() if line.strip().upper().startswith("VERDICT:")]
    verdict = verdict_line[0].split(":", 1)[1].strip() if verdict_line else "UNKNOWN"

    # Extraction de l'explication (optionnel)
    explanation_line = [line for line in response_text.splitlines() if line.strip().upper().startswith("EXPLANATION:")]
    explanation = explanation_line[0].split(":", 1)[1].strip() if explanation_line else ""

    return f"Verdict: {verdict}\nExplanation: {explanation}"


