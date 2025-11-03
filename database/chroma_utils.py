# chroma_utils.py
import numpy as np
from rag_system.azure_client import client

def get_embedding(text: str):
    """
    Génère l'embedding pour un texte via Azure OpenAI.
    """
    if not text.strip():
        return None

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def normalize_vector(vec):
    """
    Normalise un vecteur : norme = 1.
    Cela est important pour le cosine similarity et la recherche.
    """
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def cosine_similarity(vec_a, vec_b):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    vec_a = np.array(vec_a, dtype=float)
    vec_b = np.array(vec_b, dtype=float)
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
