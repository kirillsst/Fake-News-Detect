# rag_pipeline.py: Pipeline RAG complet

# from .chroma_retrieval import get_collection
# from .query_preprocess import preprocess_query # . : (imports relatifs
# from rag_system_chaima.chroma_retrieval import retrieve_chunks, get_collection
# from rag_system_chaima.query_preprocess import preprocess_query
# from rag_system_chaima.ollama_generation import generate_verdict
# from rag_system_chaima.chroma_retrieval import retrieve_chunks, get_collection

# rag_pipeline.py: Pipeline RAG complet

from .chroma_retrieval import retrieve_chunks, get_collection
from .query_preprocess import preprocess_query
from .ollama_generation import generate_verdict

def rag_pipeline(user_text: str, k: int = 5) -> str:
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
    test_text = "president donald trump on monday say he plan to nominate liberty university school of law"
    rag_pipeline(test_text)

