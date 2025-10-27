import pandas as pd
import numpy as np
from chromadb import PersistentClient
from chroma_utils import get_embedding


# Constantes

CSV_PATH = "data/processed/chunks.csv"               # Chemin vers le CSV contenant les chunks
PERSIST_DIR = "./chroma_db"           # Dossier pour la base ChromaDB persistante
COLLECTION_NAME = "fake_news_collection"
BATCH_SIZE = 50                        # Taille des lots pour l'ajout


# Fonction pour nettoyer les métadonnées

def clean_metadata(row):
    """
    Remplace les valeurs None par des chaînes vides pour ChromaDB.
    """
    metadata = {
        "label": row.get("label") if row.get("label") is not None else "",
        "article_id": row.get("article_id") if row.get("article_id") is not None else "",
        "source": row.get("source") if row.get("source") is not None else "",
        "date": row.get("date") if row.get("date") is not None else ""
    }
    return metadata


# Fonction principale d'ajout par lot

def add_chunks_to_chroma(df, client, collection_name):
    """
    Ajoute tous les chunks du DataFrame dans ChromaDB en respectant les étapes :
    - définition des métadonnées
    - génération des embeddings
    - normalisation
    - ajout à la collection
    """
    # Création ou récupération de la collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    # Initialisation des listes pour le batch
    batch_ids = []
    batch_texts = []
    batch_embeddings = []
    batch_metadatas = []

    for idx, row in df.iterrows():
        chunk_id = f"{row.get('article_id')}_{row['chunk_id']}"
        text = row["text"].strip()
        if not text:
            continue

        metadata = clean_metadata(row)

        # Génération de l'embedding
        embedding = get_embedding(text)
        if embedding is None:
            print(f"[Warning] Chunk {chunk_id} ignoré : aucun embedding disponible ou texte trop long")
            continue

        # Normalisation L2 (optionnelle)
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (embedding / norm).tolist()

        # Ajout au batch
        batch_ids.append(chunk_id)
        batch_texts.append(text)
        batch_embeddings.append(embedding)
        batch_metadatas.append(metadata)

        # Si batch plein, ajout dans ChromaDB
        if len(batch_ids) >= BATCH_SIZE:
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            print(f"[Info] {len(batch_ids)} chunks ajoutés dans la collection.")
            batch_ids, batch_texts, batch_embeddings, batch_metadatas = [], [], [], []

    # Ajouter le reste
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        print(f"[Info] {len(batch_ids)} derniers chunks ajoutés dans la collection.")


# Exécution principale

if __name__ == "__main__":
    print("[Info] Lecture du CSV...")
    df = pd.read_csv(CSV_PATH)

    print("[Info] Connexion à ChromaDB...")
    client = PersistentClient(path=PERSIST_DIR)

    print("[Info] Ajout des chunks dans la collection...")
    add_chunks_to_chroma(df, client, COLLECTION_NAME)

    print("[Info] Tous les chunks ont été ajoutés avec succès.")
