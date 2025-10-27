# rag_pipeline.py (dans rag_system)
from .query_preprocess import preprocess_query
from .chroma_retrieval import get_context_from_chroma
from .ollama_generation import generate_response
import time



def rag_analyze(user_text: str):
    start_time = time.time()
    
    clean_text = preprocess_query(user_text)
    context_text = get_context_from_chroma(clean_text)
    if not context_text.strip():
        context_text = "No relevant context found in the database."
    
    result_text = generate_response(clean_text, context_text)
    
    eval_duration = time.time() - start_time
    num_chunks = len(context_text.split())  # ou nombre réel de chunks utilisés
    
    return {
        "result_text": result_text,
        "num_chunks": num_chunks,
        "eval_duration": eval_duration
    }
if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    verdict = rag_analyze(user_text)
    print("\n=== Résultat RAG ===")
    print(verdict)
