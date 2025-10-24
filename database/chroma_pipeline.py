#chroma_pipeline.py: script principal orchestrant la pipeline
# database/chroma_pipeline.py
import pandas as pd
import numpy as np
import argparse
from database.chroma_utils import get_embedding
from chromadb import PersistentClient

# ==============================
# Constantes / valeurs par défaut
# ==============================
CSV_PATH = "data/processed/chunks.csv"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"
DEFAULT_BATCH_SIZE = 50
MAX_SUBCHUNK_WORDS = 300  # nombre de mots max par sous-chunk pour l'embedding

# ==============================
# Fonctions utilitaires
# ==============================
def split_text(text, max_words=MAX_SUBCHUNK_WORDS):
    """Divise un texte en sous-chunks de taille max_words."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def clean_metadata(row):
    """Remplace les valeurs None par des chaînes vides pour ChromaDB."""
    return {
        "label": row.get("label") or "",
        "article_id": row.get("article_id") or "",
        "source": row.get("source") or "",
        "date": row.get("date") or ""
    }

def add_chunks_to_chroma(df, client, collection_name, batch_size=DEFAULT_BATCH_SIZE):
    """Ajoute tous les chunks du DataFrame dans ChromaDB avec split et IDs uniques."""
    # Création ou récupération de la collection
    try:
        collection = client.get_collection(collection_name)
        print(f"[Info] Collection '{collection_name}' récupérée avec succès.")
    except Exception:
        collection = client.create_collection(collection_name)
        print(f"[Info] Collection '{collection_name}' créée avec succès.")

    batch_ids, batch_texts, batch_embeddings, batch_metadatas = [], [], [], []
    total_added = 0
    total_skipped = 0

    for idx, row in df.iterrows():
        article_id = row.get("article_id", idx)
        chunk_id = row.get("chunk_id", idx)
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        metadata = clean_metadata(row)

        sub_chunks = split_text(text)
        for i, sub_chunk in enumerate(sub_chunks):
            sub_chunk_id = f"{article_id}_{chunk_id}_{i}"
            embedding = get_embedding(sub_chunk)
            if embedding is None:
                print(f"[Warning] Chunk {sub_chunk_id} ignoré : aucun embedding disponible ou texte trop long")
                total_skipped += 1
                continue

            embedding = np.array(embedding)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / norm).tolist()

            batch_ids.append(sub_chunk_id)
            batch_texts.append(sub_chunk)
            batch_embeddings.append(embedding)
            batch_metadatas.append(metadata)

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                print(f"[Info] {len(batch_ids)} chunks ajoutés (total ajouté: {total_added})")
                batch_ids, batch_texts, batch_embeddings, batch_metadatas = [], [], [], []

    # Ajouter le reste
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        total_added += len(batch_ids)
        print(f"[Info] {len(batch_ids)} derniers chunks ajoutés (total ajouté: {total_added})")

    print(f"[Summary] total rows: {len(df)} | added: {total_added} | skipped: {total_skipped}")

# ==============================
# Exécution principale
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=CSV_PATH, help="Chemin vers le CSV des chunks")
    parser.add_argument("--persist-dir", type=str, default=PERSIST_DIR, help="Dossier ChromaDB")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Nom de la collection ChromaDB")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Taille des batchs")
    parser.add_argument("--max-rows", type=int, default=None, help="Nombre maximum de lignes à traiter")
    args = parser.parse_args()

    print("[Info] Lecture du CSV...")
    df = pd.read_csv(args.csv_path)
    if args.max_rows:
        df = df.head(args.max_rows)

    print(f"[Info] Connexion à ChromaDB (path={args.persist_dir})...")
    client = PersistentClient(path=args.persist_dir)

    print("[Info] Ajout des chunks dans la collection...")
    add_chunks_to_chroma(df, client, args.collection, batch_size=args.batch_size)




