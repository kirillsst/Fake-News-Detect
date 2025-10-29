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
    results = retrieve_chunks(query, collection, k=k)  # <-- récupération des chunks

    # Construction du contexte à partir des chunks récupérés avec label
    context_chunks = []
    metadatas = results["metadatas"][0]  # liste des dicts contenant 'label', 'title', etc.
    documents = results["documents"][0]

    for doc, meta in zip(documents, metadatas):
        label = meta.get("label", "Unknown")
        title = meta.get("title", "")
        # On ajoute label + titre + texte
        context_chunks.append(f"[{label}] {title}\n{doc}")

    context_text = "\n\n".join(context_chunks)  # séparation visuelle

    verdict = generate_verdict(user_text, context_text)

    print("=== Résultat RAG ===")
    print(verdict)
    print("====================\n")
    return verdict
if __name__ == "__main__":
    test_text = "reflect the good ideal of public service . for those whom I have let down , I m sorry . as I move onto the next chapter of my life , I sincerely ask for privacy for myself , my family , and my friend . sorry , mr"
    rag_pipeline(test_text)
