from chromadb import PersistentClient
from ollama._client import Client as OllamaClient
from chroma_utils import get_embedding

ollama_client = OllamaClient()

def insert_chunk(chunk_id, text, metadata, persist_dir="./chroma_db"):
    """
    Ajoute un chunk de texte avec ses métadonnées et embeddings dans ChromaDB persistante.
    """
    try:
        #  Nouveau client compatible Chroma v1.x
        client = PersistentClient(path=persist_dir)

        # Récupération ou création de la collection
        try:
            collection = client.get_collection("fake_news_collection")
        except Exception:
            collection = client.create_collection("fake_news_collection")

       # Nous obtenons un embedding déjà normalisé
        embedding = get_embedding(text)

        # Ajout dans la collection
        collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        print(f" Chunk {chunk_id} ajouté et sauvegardé dans la collection.")

    except Exception as e:
        print(f" Erreur lors de l'ajout du chunk {chunk_id} : {e}")


if __name__ == "__main__":
    insert_chunk(
        chunk_id="1",
        text="Le gouvernement a annoncé de nouvelles mesures sanitaires.",
        metadata={"source": "lemonde.fr", "date": "2025-10-20"}
    )
