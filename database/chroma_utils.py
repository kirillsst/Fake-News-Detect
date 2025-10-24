# chroma_utils.py
import numpy as np
from ollama._client import Client as OllamaClient

# Client Ollama unique pour toute l'application
ollama_client = OllamaClient()

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

def get_embedding(text: str, model: str = "all-minilm", max_words: int = 2000):
    """
    Génère l'embedding via Ollama et le normalise.
    Limite la longueur du texte pour le modèle.
    """
    try:
        # Limiter le texte par nombre de mots
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
            text = " ".join(words)
            print(f"[Warning] Texte tronqué à {max_words} mots pour l'embedding.")

        # Appel au modèle Ollama
        response = ollama_client.embeddings(model=model, prompt=text)
        embedding = response.get("embedding", None)

        if embedding is None:
            print("[Warning] Ollama a retourné None pour l'embedding.")
            return None

        # Normalisation du vecteur
        return normalize_vector(embedding)

    except Exception as e:
        print(f"[Error] Échec de génération d'embedding : {e}")
        return None

def cosine_similarity(vec_a, vec_b):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    vec_a = np.array(vec_a, dtype=float)
    vec_b = np.array(vec_b, dtype=float)
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
