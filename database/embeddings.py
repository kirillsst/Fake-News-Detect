# embeddings.py: fonctions pour générer des embeddings avec Ollama
# embeddings.py : fonctions pour générer des embeddings avec Ollama
import ollama
import numpy as np
from sympy import re

def clean_text(text: str) -> str:
    """Nettoie le texte pour éviter les caractères spéciaux et doublons d'espaces."""
    text = re.sub(r'\s+', ' ', text)  # supprime les espaces multiples
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # enlève les caractères non ASCII
    return text.strip()

def chunk_text(text, max_length=300):
    """
    Divise un texte long en morceaux de longueur fixe (par défaut 300 caractères)
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def get_embedding(text, model="nomic-embed-text"):
    """
    Récupère l'embedding d'un texte à l'aide du modèle Ollama.
    - Si le texte est trop long, il est découpé en morceaux.
    - Si plusieurs morceaux sont créés, on calcule la moyenne de leurs embeddings.
    """
    text = text.strip()
    if not text:
        return None

    # Sécurité : ignorer les textes trop longs
    if len(text) > 5000:
        print(f"[Skip] Texte trop long ({len(text)} caractères) ignoré.")
        return None

    # Réduction de la taille des chunks pour éviter l’erreur "context length exceeded"
    chunks = chunk_text(text, 500)
    embeddings = []

    for chunk in chunks:
        try:
            response = ollama.embeddings(model=model, prompt=chunk)
            emb = response.get("embedding")
            if emb:
                embeddings.append(emb)
        except Exception as e:
            print(f"[Warning] Erreur lors du traitement d’un chunk : {e}")

    if not embeddings:
        return None

    # Calcul de la moyenne des embeddings pour les chunks multiples
    avg_embedding = np.mean(np.array(embeddings), axis=0).tolist()
    return avg_embedding

# import ollama
# import numpy as np

# def chunk_text(text, max_length=500):
#     # Divise un texte long en morceaux de longueur fixe
#     return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# def get_embedding(text, model="all-minilm"):
#     """
#     Récupère l'embeddage du texte. 
#     Si le texte est trop long, il le divise en morceaux et calcule la moyenne des embeddages.
#     """
#     text = text.strip()
#     if not text:
#         return None

#     chunks = chunk_text(text, 1000)
#     embeddings = []

#     for chunk in chunks:
#         try:
#             response = ollama.embeddings(model=model, prompt=chunk)
#             emb = response.get("embedding")
#             if emb:
#                 embeddings.append(emb)
#         except Exception as e:
#             print(f"[Warning] Erreur lors du traitement du chunk : {e}")

#     if not embeddings:
#         return None

#     # nous calculons la moyenne de tous les segments dans un seul vecteur
#     avg_embedding = np.mean(np.array(embeddings), axis=0).tolist()
#     return avg_embedding