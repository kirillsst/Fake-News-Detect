from chromadb import PersistentClient

from .chroma_utils import get_embedding



client = PersistentClient(path="./chroma_db")
collection = client.get_collection("fake_news_collection")
print(f"Nombre de chunks dans la collection: {len(collection.get()['ids'])}")

query_embedding = get_embedding("war")
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
print(results)
