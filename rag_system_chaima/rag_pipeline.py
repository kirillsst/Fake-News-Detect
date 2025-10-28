# rag_pipeline.py: Pipeline RAG complet

from .chroma_retrieval import retrieve_chunks, get_collection
from .query_preprocess import preprocess_query
from .ollama_generation import generate_verdict

def rag_pipeline(user_text: str, k: int = 10) -> str:
    """
    Pipeline complet :
    1. Prétraitement du texte utilisateur
    2. Recherche des chunks pertinents dans ChromaDB
    3. Génération du verdict TRUE/FAKE + justification via Ollama
    """
    query = preprocess_query(user_text)
    collection = get_collection()
    context = retrieve_chunks(query, collection, k=k)
    verdict = generate_verdict(user_text, context)
    
    print("=== Résultat RAG ===")
    print(verdict)
    print("====================\n")
    return verdict

# Exemple d'utilisation
if __name__ == "__main__":
    test_text = "florida after the christmas holiday . the tax package , the large such overhaul since the 1980s"
    rag_pipeline(test_text)