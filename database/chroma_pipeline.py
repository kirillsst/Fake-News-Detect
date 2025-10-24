#chroma_pipeline.py: script principal orchestrant la pipeline

# database/chroma_pipeline.py
"""
Pipeline d'ingestion dans ChromaDB
- lit data/processed/chunks.csv
- génère embeddings (via chroma_utils.get_embedding ou embeddings.get_embedding)
- insère par batch dans la collection 'fake_news_collection'
Usage:
    python -m database.chroma_pipeline
    python -m database.chroma_pipeline --max-rows 100 --batch-size 32
"""

import os
import argparse
import math
import pandas as pd
import numpy as np
import chromadb
from chromadb import PersistentClient

# Essayer d'importer get_embedding depuis chroma_utils, fallback vers embeddings
try:
    from database.chroma_utils import get_embedding
except Exception:
    try:
        from database.embeddings import get_embedding
    except Exception:
        get_embedding = None

# Constantes / valeurs par défaut
CSV_PATH = "data/processed/chunks.csv"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "fake_news_collection"
DEFAULT_BATCH_SIZE = 50

def ensure_collection(client: PersistentClient, name: str):
    """Crée ou récupère la collection ChromaDB."""
    try:
        collection = client.get_collection(name=name)
        print(f"[Info] Collection '{name}' récupérée avec succès.")
    except Exception:
        collection = client.create_collection(name=name)
        print(f"[Info] Collection '{name}' créée.")
    return collection

def normalize_vector(vec):
    """Normalise un vecteur L2 et retourne une liste (prête pour Chroma)."""
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def validate_df(df: pd.DataFrame):
    """
    Normalise les noms de colonnes attendues : accepte 'text' ou 'chunk_text' comme champ texte.
    Retourne dataframe avec colonnes: 'article_id','chunk_id','text','label' (ou lève erreur).
    """
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    # colonnes possibles pour le texte
    if "text" in df.columns:
        text_col = "text"
    elif "chunk_text" in df.columns:
        text_col = "chunk_text"
        df = df.rename(columns={"chunk_text": "text"})
    else:
        raise ValueError("Aucune colonne 'text' ou 'chunk_text' trouvée dans le CSV.")

    # s'assurer que les colonnes essentielles existent
    for c in ("article_id", "chunk_id"):
        if c not in df.columns:
            # si pas présentes, on génère des identifiants basiques
            print(f"[Warn] Colonne '{c}' non trouvée → génération automatique.")
            if c == "article_id":
                df["article_id"] = df.index  # fallback
            else:
                df["chunk_id"] = 0

    # remplir les labels manquants par ''
    if "label" not in df.columns:
        df["label"] = ""
    df["text"] = df["text"].fillna("").astype(str)

    return df[["article_id", "chunk_id", "text", "label"]]

def add_chunks_to_chroma(df: pd.DataFrame, client: PersistentClient, collection_name: str, batch_size: int = DEFAULT_BATCH_SIZE):
    """Ajout en batch dans ChromaDB avec génération d'embeddings."""
    if get_embedding is None:
        raise RuntimeError("Aucune fonction get_embedding disponible. Vérifie database/chroma_utils.py ou database/embeddings.py")

    collection = ensure_collection(client, collection_name)

    total = len(df)
    added = 0
    skipped = 0

    batch_ids = []
    batch_texts = []
    batch_embeddings = []
    batch_metadatas = []

    for idx, row in df.iterrows():
        chunk_id = f"{row['article_id']}_{row['chunk_id']}"
        text = row["text"].strip()
        if not text:
            skipped += 1
            continue

        # Génération embedding (peut retourner None si texte trop long ou erreur)
        embedding = None
        try:
            embedding = get_embedding(text)
        except Exception as e:
            print(f"[Error] Erreur get_embedding pour chunk {chunk_id} : {e}")
            embedding = None

        if embedding is None:
            print(f"[Warning] Chunk {chunk_id} ignoré : aucun embedding disponible ou texte trop long")
            skipped += 1
            continue

        # normalisation L2 (sécurisée)
        try:
            embedding = normalize_vector(embedding)
        except Exception as e:
            print(f"[Error] Normalisation échouée pour chunk {chunk_id} : {e}")
            skipped += 1
            continue

        metadata = {
            "label": row.get("label", "") if pd.notna(row.get("label", "")) else "",
            "article_id": row.get("article_id", ""),
        }

        batch_ids.append(chunk_id)
        batch_texts.append(text)
        batch_embeddings.append(embedding)
        batch_metadatas.append(metadata)

        # flush batch
        if len(batch_ids) >= batch_size:
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                added += len(batch_ids)
                print(f"[Info] {len(batch_ids)} chunks ajoutés (total ajouté: {added}).")
            except Exception as e:
                print(f"[Error] Échec ajout batch : {e}")
            batch_ids, batch_texts, batch_embeddings, batch_metadatas = [], [], [], []

    # ajouter le reste si présent
    if batch_ids:
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            added += len(batch_ids)
            print(f"[Info] {len(batch_ids)} derniers chunks ajoutés (total ajouté: {added}).")
        except Exception as e:
            print(f"[Error] Échec ajout dernier batch : {e}")

    print(f"\n[Summary] total rows: {total} | added: {added} | skipped: {skipped}")

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline d'import ChromaDB")
    p.add_argument("--csv", type=str, default=CSV_PATH, help="Chemin vers chunks.csv")
    p.add_argument("--persist", type=str, default=PERSIST_DIR, help="Répertoire ChromaDB persist")
    p.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Nom de la collection")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Taille des batches pour insert")
    p.add_argument("--max-rows", type=int, default=None, help="Limiter le nombre de lignes lues (pour test)")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"{args.csv} non trouvé. Génère-le avec le pipeline de prétraitement.")

    print("[Info] Lecture du CSV...")
    df = pd.read_csv(args.csv)

    if args.max_rows:
        df = df.head(args.max_rows)

    try:
        df = validate_df(df)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la validation du CSV : {e}")

    print(f"[Info] Connexion à ChromaDB (path={args.persist})...")
    client = PersistentClient(path=args.persist)

    print("[Info] Ajout des chunks dans la collection...")
    add_chunks_to_chroma(df, client, args.collection, batch_size=args.batch_size)

if __name__ == "__main__":
    main()


