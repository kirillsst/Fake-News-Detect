# chroma_setup: création client + collection

#version KKiril : database/chroma_setup.py
import chromadb
from chromadb.config import Settings

# Création du client ChromaDB
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Nom de la collection
collection_name = "fake_news_collection"

# Créer ou récupérer la collection
try:
    collection = client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' récupérée avec succès !")
except Exception as e:  # Remplace CollectionNotFoundError
    print(f"Collection '{collection_name}' non trouvée. Création d'une nouvelle collection...")
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' créée avec succès !")


# import chromadb
# from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# def create_client_and_collection(collection_name, embedding_model="nomic-embed-text", db_path="./chroma_db"):
#     """
#     Crée le client ChromaDB et récupère ou crée la collection
#     """
#     client = chromadb.PersistentClient(path=db_path)
#     collection = client.get_or_create_collection(name=collection_name)
#     embedding_fn = OllamaEmbeddingFunction(model_name=embedding_model)
#     return client, collection, embedding_fn

# import chromadb
# from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# def create_client_and_collection(collection_name, embedding_model="nomic-embed-text", db_path="./chroma_db"):
#     """
#     Crée le client ChromaDB et récupère ou crée la collection
#     """
#     # Initialisation du client
#     client = chromadb.PersistentClient(path=db_path)

#     # Récupérer ou créer la collection
#     collection = client.get_or_create_collection(name=collection_name)

#     # Fonction d'embedding
#     embedding_fn = OllamaEmbeddingFunction(model_name=embedding_model)

#     return client, collection, embedding_fn
