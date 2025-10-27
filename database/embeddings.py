import ollama
import numpy as np

def chunk_text(text, max_words=130):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def get_embedding(text, model="all-minilm"):
    """
    Récupère l'embeddage du texte. 
    Si le texte est trop long, il le divise en morceaux et calcule la moyenne des embeddages.
    """
    text = text.strip()
    if not text:
        return None

    chunks = chunk_text(text, max_words=150)
    embeddings = []

    for chunk in chunks:
        try:
            response = ollama.embeddings(model=model, prompt=chunk)
            emb = response.get("embedding")
            if emb:
                embeddings.append(emb)
        except Exception as e:
            print(f"[Warning] Erreur lors du traitement du chunk : {e}")

    if not embeddings:
        return None

    # nous calculons la moyenne de tous les segments dans un seul vecteur
    avg_embedding = np.mean(np.array(embeddings), axis=0).tolist()
    return avg_embedding