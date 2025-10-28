from .query_preprocess import preprocess_query
from .chroma_retrieval import get_context_from_chroma
from .ollama_generation import generate_response
import time
import pandas as pd

CSV_PATH = "data/processed/chunks.csv"

def get_label_from_csv(user_text: str) -> str:
    """
    Retourne le label (TRUE/FAKE) pour le texte donné
    en comparant avec le CSV des chunks/articles.
    """
    df = pd.read_csv(CSV_PATH)
    # Cherche le texte exact ou partiel dans la colonne 'text'
    match = df[df['text'].str.contains(user_text, case=False, na=False)]
    if not match.empty:
        return match.iloc[0]['label'].upper()
    return None

def check_chunk_consistency(verdict: str, chunks_metadata: list) -> bool:
    """
    Vérifie si la majorité des chunks supporte le verdict.
    verdict : "TRUE" ou "FAKE"
    chunks_metadata : liste de dict contenant 'label'
    """
    true_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() == "TRUE")
    false_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() == "FALSE")

    if true_count + false_count == 0:
        return False  # pas de label disponible

    majority_label = "TRUE" if true_count >= false_count else "FAKE"
    return verdict.upper() == majority_label

def rag_analyze(user_text: str, max_chunks: int = 5, max_chars: int = 1000):
    start_time = time.time()

    # 1️⃣ Prétraitement du texte utilisateur
    clean_text = preprocess_query(user_text[:max_chars])

    # 2️⃣ Récupération du contexte via Chroma (texte + métadatas)
    query_start = time.time()
    context_text, context_metadatas = get_context_from_chroma(clean_text, n_results=max_chunks)
    print(f"Temps requête Chroma : {time.time() - query_start:.2f} sec")

    if not context_text.strip():
        context_text = "No relevant context found in the database."

    # 3️⃣ Génération de la réponse avec Ollama
    ollama_start = time.time()
    result_text = generate_response(clean_text, context_text)
    print(f"Temps génération Ollama : {time.time() - ollama_start:.2f} sec")

    # 4️⃣ Extraction du verdict
    verdict = "UNKNOWN"
    for line in result_text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()

    # 5️⃣ Vérification cohérence verdict / chunks
    chunks_consistency = check_chunk_consistency(verdict, context_metadatas)

    # 6️⃣ Récupération automatique du label réel depuis CSV
    ground_truth_label = get_label_from_csv(user_text)

    # 7️⃣ Temps total et nombre de chunks
    eval_duration = time.time() - start_time
    num_chunks = len(context_text.split())  # indicatif

    # 8️⃣ Affichage terminal
    print("\n=== Résultat RAG ===")
    print(f"Verdict : {verdict}")
    if ground_truth_label:
        print(f"Label réel : {ground_truth_label}")
    print(f"Cohérence avec les chunks : {' Oui' if chunks_consistency else ' Non'}")
    print(f"Nombre de chunks utilisés : {num_chunks}")
    print(f"Durée totale : {eval_duration:.2f} sec")
    print(f"\nExplication et contexte :\n{result_text}\n")

    return {
        "result_text": result_text,
        "verdict": verdict,
        "ground_truth_label": ground_truth_label,
        "chunks_consistency": chunks_consistency,
        "num_chunks": num_chunks,
        "eval_duration": eval_duration
    }


if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    rag_analyze(user_text)
