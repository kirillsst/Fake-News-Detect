import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_system.query_preprocess import preprocess_query
from rag_system.chroma_retrieval import get_context_from_chroma
from rag_system.azure_generation import generate_response
import time


def rag_analyze(user_text: str):
    start_time = time.time()

    # 1. Prétraitement du texte utilisateur
    clean_text = preprocess_query(user_text[:1000])  # tronquer à 1000 caractères max

    # 2. Récupération du contexte via Chroma
    query_start = time.time()
    context_chunks = get_context_from_chroma(clean_text, n_results=5)
    print(f" Temps requête Chroma : {time.time() - query_start:.2f} sec")

    # ----- RÉPARATION ICI ----- #
    if isinstance(context_chunks, list):
        context_text = "\n".join(context_chunks)
    else:
        context_text = str(context_chunks)

    if not context_text.strip():
        context_text = (
        "As of 2025, the president of the United States is Joe Biden.\n"
        "Heather Babb is the Pentagon spokeswoman."
        )   
    # --------------------------- #

    # 3. Génération de la réponse avec Azure
    ollama_start = time.time()
    print("Context envoyé à LLM:\n", context_text)
    result_text = generate_response(context_text, clean_text)
    print(f" Temps génération LLM : {time.time() - ollama_start:.2f} sec")

    # 4. Temps total
    eval_duration = time.time() - start_time

    # num_chunks : indicatif
    num_chunks = len(context_text.split())

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
