import chromadb
from database.chroma_utils import get_embedding

# Connexion à la base Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("fake_news_collection")


def get_context_from_chroma(query: str, n_results: int = 5):
    """
    Retourne les documents les plus similaires dans ChromaDB.
    """
    embedding = get_embedding(query)
    if embedding is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[ERREUR CHROMA] {e}")
        return []
