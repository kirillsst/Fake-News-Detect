import re
import time
import pandas as pd
from .query_preprocess import preprocess_query
from .chroma_retrieval import get_context_from_chroma
from .ollama_generation import generate_response

CSV_PATH = "data/processed/chunks.csv"


def get_label_from_csv(user_text: str) -> str:
    """Retourne le label majoritaire (TRUE/FAKE) parmi les lignes du CSV dont le texte contient la requête utilisateur."""
    df = pd.read_csv(CSV_PATH)
    
    query = user_text.strip()[:80]
    if not query:
        return None

    subset = df[df['text'].str.contains(query, case=False, na=False, regex=False)]
    if subset.empty:
        return None

    counts = subset['label'].str.upper().value_counts()
    if "FAKE" in counts and "TRUE" in counts:
        return "FAKE" if counts["FAKE"] > counts["TRUE"] else "TRUE"
    elif "FAKE" in counts:
        return "FAKE"
    elif "TRUE" in counts:
        return "TRUE"
    else:
        return None


def check_chunk_consistency(verdict: str, chunks_metadata: list) -> bool:
    """Vérifie si la majorité des chunks supporte le verdict."""
    true_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() == "TRUE")
    fake_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() in ["FAKE", "FALSE"])
    if true_count + fake_count == 0:
        return False
    majority_label = "TRUE" if true_count > fake_count else "FAKE"
    return verdict.upper() == majority_label


def final_verdict(verdict_llm, chunks_metadata):
    """
    Combine verdict LLM et majorité des chunks.
    En cas de divergence, on privilégie FAKE pour éviter les faux positifs.
    """
    true_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() == "TRUE")
    fake_count = sum(1 for m in chunks_metadata if str(m.get("label")).upper() in ["FAKE", "FALSE"])

    if true_count + fake_count == 0:
        return verdict_llm

    majority_chunks = "TRUE" if true_count >= fake_count else "FAKE"

    if majority_chunks != verdict_llm:
        if "FAKE" in [majority_chunks, verdict_llm]:
            return "FAKE"
        else:
            return verdict_llm
    return verdict_llm


def rag_analyze(user_text: str, max_chunks: int = 20, max_chars: int = 1000):
    """Pipeline complet RAG : prétraitement, récupération du contexte, génération et verdict final."""
    start_time = time.time()

    # --- 1️ Prétraitement ---
    clean_text = preprocess_query(user_text[:max_chars])

    # --- 2️ Récupération du contexte depuis Chroma ---
    query_start = time.time()
    context_text, context_metadatas = get_context_from_chroma(clean_text, n_results=max_chunks)
    print(f"Temps requête Chroma : {time.time() - query_start:.2f} sec")

    if not context_text.strip():
        context_text = "No relevant context found in the database."

    # --- 3️ Réponse Ollama ---
    ollama_start = time.time()
    result_text = generate_response(clean_text, context_text, context_metadatas)
    print(f"Temps génération Ollama : {time.time() - ollama_start:.2f} sec")

    # --- 4️ Extraction du verdict avec regex robuste ---
    verdict = "UNKNOWN"
    matches = re.findall(r'\bVERDICT[:\-]?\s*(TRUE|FAKE|FALSE)', result_text, flags=re.IGNORECASE)
    if matches:
        verdict = matches[-1].upper()  # dernier verdict trouvé = plus fiable
    else:
        # fallback : recherche d’un mot isolé TRUE/FAKE si pas de ligne VERDICT
        matches = re.findall(r'\b(TRUE|FAKE|FALSE)\b', result_text, flags=re.IGNORECASE)
        verdict = matches[-1].upper() if matches else "UNKNOWN"

    # Uniformisation du verdict
    if verdict == "FALSE":
        verdict = "FAKE"
    elif verdict not in ["TRUE", "FAKE"]:
        verdict = "FAKE"  # ⚠️ on force "FAKE" si doute (évite les faux positifs TRUE)

    # --- 5️   Combinaison verdict LLM + chunks ---
    verdict = final_verdict(verdict, context_metadatas)

    # --- 6️ Récupération du label réel ---
    ground_truth_label = get_label_from_csv(user_text)

    # --- 7️ Mise à jour du texte de sortie ---
    result_lines = result_text.splitlines()
    for i, line in enumerate(result_lines):
        if line.strip().upper().startswith("VERDICT"):
            result_lines[i] = f"VERDICT: {verdict}"
    result_text = "\n".join(result_lines)

    # --- 8️ Statistiques et affichage ---
    eval_duration = time.time() - start_time
    num_chunks = len(context_metadatas)

    print("\n=== Résultat RAG ===")
    print(f"Verdict : {verdict}")
    if ground_truth_label:
        print(f"Label réel : {ground_truth_label}")
    print(f"Cohérence avec les chunks : {'Oui' if check_chunk_consistency(verdict, context_metadatas) else 'Non'}")
    print(f"Nombre de chunks utilisés : {num_chunks}")
    print(f"Durée totale : {eval_duration:.2f} sec")
    print(f"\nExplication et contexte :\n{result_text}\n")

    # --- 9️ Résultat retourné pour évaluation ---
    return {
        "result_text": result_text,
        "verdict": verdict,
        "ground_truth_label": ground_truth_label,
        "chunks_consistency": check_chunk_consistency(verdict, context_metadatas),
        "num_chunks": num_chunks,
        "eval_duration": eval_duration
    }


if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    rag_analyze(user_text)
