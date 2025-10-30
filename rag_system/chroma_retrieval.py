# Recherche de documents similaires dans ChromaDB
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chromadb import PersistentClient
from database.chroma_utils import get_embedding

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"

def get_context_from_chroma(user_text: str, n_results: int = 5):
    """
    Récupère l'intégration de la requête, recherche les chunks les plus proches et renvoie le contexte + métadonnées.
    """
    query_embedding = get_embedding(user_text)
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    context_chunks = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_chunks.append(f"[{metadata['label'].upper()}] {doc}")

    context_text = " ".join(context_chunks)
    context_metadatas = results["metadatas"][0]  # ✅ ajout ici

    return context_text, context_metadatas

