# rag_system/rag_core.py
import time
from .query_preprocess import preprocess_query
from .chroma_retrieval import get_context_from_chroma
from .ollama_generation import generate_response

def rag_analyze(user_text: str, max_chunks: int = 20, max_chars: int = 1000):
    start_time = time.time()

    clean_text = preprocess_query(user_text[:max_chars])
    context_text, context_metadatas = get_context_from_chroma(clean_text, n_results=max_chunks)

    if not context_text.strip():
        context_text = "No relevant context found in the database."

    result_text = generate_response(clean_text, context_text)
    eval_duration = time.time() - start_time
    num_chunks = len(context_metadatas)

    return {
        "result_text": result_text,
        "context_metadatas": context_metadatas,
        "num_chunks": num_chunks,
        "eval_duration": eval_duration
    }
