"""
Étape 2 — Stockage des données dans ChromaDB
--------------------------------------------
Ce script charge les chunks nettoyés depuis un CSV, génère leurs embeddings
via Ollama, puis les stocke dans une collection ChromaDB avec métadonnées.
"""

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# =========  1. PARAMÈTRES ==========
CSV_PATH = "data/articles_chunks.csv"          # chemin vers ichier CSV
COLLECTION_NAME = "fake_news_chunks"           # nom de la collection Chroma
EMBEDDING_MODEL = "nomic-embed-text"           # modèle d'embedding Ollama

# =========  2. INITIALISATION ==========

# Client local ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")  # dossier de stockage local

# Charger ou créer la collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Fonction d'embedding (avec Ollama)
embedding_fn = OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL)


# =========  3. CHARGEMENT DU DATAFRAME ==========
print("Chargement des données depuis :", CSV_PATH)
sample_df = pd.read_csv(CSV_PATH)

# Vérification rapide
print(f"{len(sample_df)} lignes chargées.")
print(sample_df.head(2))


# =========  4. AJOUT DES CHUNKS ==========
print(f"Ajout des chunks dans la collection '{COLLECTION_NAME}'...")

for idx, row in sample_df.iterrows():
    chunk_id = f"{row['article_id']}_{row['chunk_id']}"

    metadata = {
        "title": row.get("title", ""),
        "label": row["label"],
        "article_id": int(row["article_id"]),
    }

    try:
        collection.add(
            ids=[chunk_id],
            documents=[row["chunk_text"]],
            metadatas=[metadata],
            embeddings=embedding_fn([row["chunk_text"]]),
        )
    except Exception as e:
        print(f"Erreur lors de l’ajout du chunk {chunk_id} : {e}")

print(f"{len(sample_df)} chunks ajoutés à la collection '{COLLECTION_NAME}'.")
print("Stockage terminé avec succès !")
