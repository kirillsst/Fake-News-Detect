from chromadb import PersistentClient

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"

# Connexion à la base
client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Vérification du nombre total de chunks / embeddings
total_embeddings = len(collection.get("embeddings"))
print(f"Nombre total d'embeddings dans la collection : {total_embeddings}")
