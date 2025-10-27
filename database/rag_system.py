# rag_system_concret.py
# ----------------------------
# Système RAG pour analyser un article via ChromaDB et Ollama
# Entrée : texte utilisateur
# Sortie : verdict TRUE/FAKE avec justification
# ----------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ollama
from chromadb import PersistentClient
from process_data.preprocessing_article import PreprocessingArticle
from chroma_utils import get_embedding
import sys


# ----------------------------
# Configuration
# ----------------------------
CHROMA_DB_PATH = "./chroma_db"          # chemin vers la base ChromaDB
COLLECTION_NAME = "fake_news_collection"
MODEL_NAME = "phi3:mini"                # modèle léger pour usage local
N_RESULTS = 5                            # nombre de chunks similaires à récupérer

# ----------------------------
# Fonction pour récupérer le contexte depuis ChromaDB
# ----------------------------
def get_context_from_chroma(query_embedding, n_results=N_RESULTS):
    """
    Cherche les chunks les plus proches dans ChromaDB
    et assemble le contexte pour le LLM.
    """
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Concaténer les textes des chunks récupérés
    context_chunks = results['documents'][0]  # liste de strings
    context_text = " ".join(context_chunks)
    
    # Retourner le texte du contexte
    return context_text

# ----------------------------
# Fonction principale RAG
# ----------------------------
def rag_analyze(user_text):
    """
    Analyse un texte utilisateur en utilisant la base ChromaDB
    et le modèle Ollama pour raisonner par comparaison.
    """
    # Prétraitement du texte (pas de chunk)
    article = PreprocessingArticle(user_text, lemmatize=True)
    clean_text = article.clean_text()
    if article.lemmatize:
        clean_text = article.lemmatize_text()
    
    # Vectorisation / embedding
    query_embedding = get_embedding(clean_text)
    if query_embedding is None or len(query_embedding) == 0:
        print("[Erreur] Impossible de générer l'embedding pour ce texte.")
        return
    
    # Recherche des chunks les plus proches
    context_text = get_context_from_chroma(query_embedding)
    
    # Formulation du prompt pour le modèle
    prompt = f"""
You are a fact-checker AI. Using the following context from news articles, 
analyze the user's text and determine whether it is TRUE or FAKE.
Provide a verdict (TRUE or FAKE) and a short explanation.

User Text:
{clean_text}

Context from similar articles:
{context_text}
"""
    
    # Appel au modèle Ollama
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a fact-checking AI."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Retourner le verdict et l'explication
    return response

# ----------------------------
# Exécution directe
# ----------------------------
if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    verdict = rag_analyze(user_text)
    print("\n=== Résultat RAG ===")
    print(verdict)
