
from chromadb import PersistentClient
from database.chroma_utils import get_embedding

CSV_PATH = "data/processed/chunks.csv"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"

def get_context_from_chroma(user_text: str, n_results: int = 5) -> str:
    """
    Récupère l'intégration de la requête, recherche les chunks les plus proches et renvoie le contexte combiné.
    """
    query_embedding = get_embedding(user_text)
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    context_chunks = results["documents"][0]
    return " ".join(context_chunks)
