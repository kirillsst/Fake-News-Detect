# Recherche de documents similaires dans ChromaDB
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chromadb import PersistentClient
from database.chroma_utils import get_embedding

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"

def get_context_from_chroma(user_text: str, n_results: int = 5) -> str:
    """
    Récupère l'intégration de la requête, recherche les chunks les plus proches et renvoie le contexte combiné.
    """
    if not user_text or not user_text.strip():
        raise ValueError("Le texte fourni est vide. Impossible de générer des embeddings.")

    # Génération de l'embedding
    query_embedding = get_embedding(user_text)
    if query_embedding is None:
        raise ValueError("Impossible de générer l'embedding pour le texte fourni. Vérifie ta clé API ou le modèle d'embeddings.")

    # Conversion en liste si c'est un numpy array
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Requête dans ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Construction du contexte combiné
    context_chunks = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_chunks.append(f"[{metadata['label'].upper()}] {doc}")

    context_text = " ".join(context_chunks)
    return context_text


# Test rapide en local
if __name__ == "__main__":
    sample_text = "fake news example"
    context = get_context_from_chroma(sample_text)
    print("Contexte récupéré :", context)
