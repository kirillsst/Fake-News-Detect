from ollama._client import Client as OllamaClient
from .chroma_setup import get_chroma_collection
from .chroma_utils import normalize_vector

ollama_client = OllamaClient()

def insert_chunk(chunk_id, text, metadata, collection):
    try:
        embedding_response = ollama_client.embeddings(model="all-minilm", prompt=text)
        embedding = normalize_vector(embedding_response["embedding"])

        collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        print(f" Chunk {chunk_id} ajouté.")
    except Exception as e:
        print(f" Erreur avec le chunk {chunk_id} : {e}")
