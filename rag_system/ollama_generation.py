import ollama

def generate_response(user_text: str, context_text: str, context_metadatas: list) -> str:
    """
    Génère une réponse à partir du texte utilisateur et du contexte récupéré via Chroma.
    Le modèle doit retourner un format strictement conforme pour être analysé correctement.
    """
    prompt = f"""
Tu es un assistant de vérification de faits (fact-checker).
Ton rôle est de déterminer si le texte donné est VRAI ou FAUX à partir du contexte.

Consignes :
- Analyse attentivement le texte et le contexte.
- Ne fais aucune supposition sans base factuelle.
- À la fin de ta réponse, tu dois ABSOLUMENT conclure par une ligne unique au format :

VERDICT: TRUE
ou
VERDICT: FAKE

- Ne mets rien après cette ligne.
- Si tu hésites, choisis FAKE (pour éviter les faux positifs).
- Ajoute une explication claire juste avant le verdict, au format :

EXPLANATION: <raison concise>

Texte utilisateur :
{user_text}

Contexte :
{context_text}

Réponds UNIQUEMENT dans ce format exact :

EXPLANATION: <ta réponse ici>
VERDICT: <TRUE ou FAKE>
"""

    response = ollama.generate(
        model="mistral:latest",
        prompt=prompt,
        options={"temperature": 0.2}  # faible température = plus de cohérence
    )

    return response["response"]
