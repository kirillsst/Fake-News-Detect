# chroma_retrieval.py: Recherche des chunks similaires dans ChromaDB
# rag_system_chaima/chroma_retrieval.py
from chromadb import Client
from chromadb.utils import embedding_functions
from chromadb import PersistentClient

def get_client():
    """
    Crée un client ChromaDB persistant (nouvelle API Chroma >= 0.5).
    Les données seront sauvegardées dans le dossier ./chroma_db/
    """
    client = PersistentClient(path="./chroma_db")
    return client

def get_collection(collection_name="fake_news_chunks"):
    """
    Récupère ou crée une collection ChromaDB avec la fonction d'embedding Ollama.
    """
    client = get_client()

    # Vérifie si la collection existe déjà
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' récupérée.")
    else:
        # Définir la fonction d'embedding Ollama
        embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name="all-minilm")

        # Création de la collection
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        print(f"Collection '{collection_name}' créée.")

    return collection

def retrieve_chunks(user_query, collection, k=5):
    """
    Recherche les k chunks les plus proches de user_query dans la collection.
    Retourne un dict avec 'documents' et 'metadatas'.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=k
    )
    return results

def add_chunks_to_collection(collection, df_chunks, max_chunk_length=512, batch_size=50):
    """
    Ajoute des chunks depuis un DataFrame dans la collection ChromaDB.
    """
    # Récupérer les ids existants
    existing_ids = set()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    for m in all_meta:
        if isinstance(m, list):
            for item in m:
                if isinstance(item, dict) and "id" in item:
                    existing_ids.add(item["id"])
    print(f"{len(existing_ids)} chunks déjà présents dans la collection.")

    from tqdm import tqdm

    for start in tqdm(range(0, len(df_chunks), batch_size), desc="Ajout chunks"):
        batch = df_chunks.iloc[start:start+batch_size]

        # Générer les ids
        ids = [f"{row['article_id']}_{row['chunk_id']}" for _, row in batch.iterrows()]
        # Filtrer les chunks déjà existants
        new_idx = [i for i, cid in enumerate(ids) if cid not in existing_ids]
        if not new_idx:
            continue

        batch = batch.iloc[new_idx]
        ids = [f"{row['article_id']}_{row['chunk_id']}" for _, row in batch.iterrows()]

        # Préparer metadatas
        metadatas = [
            {
                "title": row.get("title", ""),
                "label": row["label"],
                "article_id": int(row["article_id"]),
                "id": ids[i]
            }
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        # Tronquer les chunks trop longs
        texts = [text[:max_chunk_length] for text in batch["chunk_text"].tolist()]

        # Calcul des embeddings
        embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name="all-minilm")
        try:
            embeddings = embedding_fn(texts)
        except Exception as e:
            print(f"⚠️ Erreur embedding pour ce batch ({start}-{start+batch_size}): {e}")
            continue

        # Ajout dans la collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    print("✅ Tous les chunks ajoutés (ou ignorés si déjà présents).")
    