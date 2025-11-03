# chroma_utils.py
import os
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

# Charger les variables du .env
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview"
)

def normalize_vector(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def get_embedding(text: str):
    """
    Génère l'embedding du texte via Azure OpenAI SDK.
    """
    if not text or not text.strip():
        print("[Erreur] Texte vide, impossible de générer un embedding.")
        return None

    try:
        response = client.embeddings.create(
            model=deployment,  # Utilisation correcte de la variable
            input=text
        )
        embedding = response.data[0].embedding
        return normalize_vector(embedding)

    except Exception as e:
        print(f"[ERREUR EMBEDDING] Impossible de générer l'embedding : {e}")
        return None

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a, dtype=float)
    vec_b = np.array(vec_b, dtype=float)
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
