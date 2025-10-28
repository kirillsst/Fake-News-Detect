# rag_pipeline.py : Pipeline principal du système RAG
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_system.query_preprocess import preprocess_query
from rag_system.chroma_retrieval import get_context_from_chroma
from rag_system.ollama_generation import generate_response

def rag_analyze(user_text: str):
    """
    Combine toutes les étapes : nettoyage, recherche dans la base et génération de la réponse.
    """
    # Prétraitement
    clean_text = preprocess_query(user_text)

    # Recherche contextuelle
    context_text = get_context_from_chroma(clean_text)

    # Génération de la réponse
    result = generate_response(clean_text, context_text)

    return result

if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    verdict = rag_analyze(user_text)
    print("\n=== Résultat RAG ===")
    print(verdict)