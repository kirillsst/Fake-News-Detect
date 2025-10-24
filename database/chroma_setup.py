from chromadb import PersistentClient

def get_chroma_collection(collection_name="fake_news_collection", persist_dir="./chroma_db"):
    # Création du client persistant
    client = PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' récupérée avec succès !")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' créée avec succès !")

    return collection

if __name__ == "__main__":
    collection = get_chroma_collection()
    print(" Setup terminé")