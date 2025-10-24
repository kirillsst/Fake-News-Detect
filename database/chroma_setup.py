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
