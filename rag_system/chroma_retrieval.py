from chromadb import PersistentClient
from database.chroma_utils import get_embedding

# Constantes
CSV_PATH = "data/processed/chunks.csv"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"


def get_context_from_chroma(user_text: str, n_results: int = 20):
    """
    Récupère l'embedding de la requête utilisateur,
    recherche les chunks les plus proches dans ChromaDB,
    et renvoie :
        - context_text : texte combiné des chunks pour le prompt
        - context_metadatas : liste des métadonnées de chaque chunk, incluant le label
    """
    try:
        # 1️⃣ Calcul de l'embedding du texte utilisateur
        query_embedding = get_embedding(user_text)

        # 2️⃣ Connexion à la base Chroma persistante
        client = PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        # 3️⃣ Recherche des chunks les plus proches
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # 4️⃣ Récupération des textes et métadonnées
        context_chunks = results.get("documents", [[]])[0]
        context_metadatas = results.get("metadatas", [[]])[0]  # <- labels ici

        if not context_chunks:
            print(" Aucun contexte pertinent trouvé dans ChromaDB.")
            return "", []

        print(f" {len(context_chunks)} chunks récupérés depuis ChromaDB.")

        # 5️⃣ Fusion propre du texte pour le prompt
        context_text = "\n---\n".join(context_chunks)

        return context_text, context_metadatas

    except Exception as e:
        print(f" Erreur lors de la recherche Chroma : {e}")
        return "", []
