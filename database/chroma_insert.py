# chroma_retrieval.py
from chromadb import PersistentClient
from .chroma_utils import get_embedding

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"

def get_context_from_chroma(user_text: str, n_results: int = 10, max_context_words: int = 500) -> str:
    """
    Récupère l'intégration de la requête, recherche les chunks les plus proches et renvoie
    le contexte combiné, limité en longueur pour le modèle.
    
    Args:
        user_text (str): Texte utilisateur à analyser.
        n_results (int): Nombre de chunks les plus proches à récupérer.
        max_context_words (int): Nombre maximum de mots dans le contexte final.

    Returns:
        str: Contexte combiné des chunks les plus pertinents.
    """
    # Génération de l'embedding du texte utilisateur
    query_embedding = get_embedding(user_text)
    if query_embedding is None:
        print("[Warning] Impossible de générer l'embedding du texte utilisateur.")
        return ""

    # Connexion à ChromaDB
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Recherche des n chunks les plus proches
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    context_chunks = results["documents"][0]  # Liste de textes récupérés

    if not context_chunks:
        print("[Info] Aucun chunk pertinent trouvé dans la collection.")
        return ""

    # Concaténer les chunks et limiter le nombre de mots pour le modèle
    context_text = " ".join(context_chunks)
    words = context_text.split()
    if len(words) > max_context_words:
        context_text = " ".join(words[:max_context_words])
        print(f"[Info] Contexte tronqué à {max_context_words} mots pour le modèle.")

    print(f"[Info] {len(context_chunks)} chunks récupérés depuis ChromaDB.")
    return context_text
