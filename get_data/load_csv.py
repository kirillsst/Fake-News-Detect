import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk

# --- Téléchargement des ressources NLTK ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

RAW_PATH = "data/raw"         
PROCESSED_PATH = "data/processed"  

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Fonction de nettoyage du texte ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# --- Fonction de chunking avec chevauchement ---
def chunk_text_overlap(text, max_words=100, overlap=20):
    if not isinstance(text, str) or text.strip() == "":
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

# --- Liste tous les fichiers CSV dans RAW_PATH ---
csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(RAW_PATH, file_name)
    print(f"\n--- Traitement du fichier : {file_name} ---")
    
    # Lecture CSV
    df = pd.read_csv(file_path)
    print(f"Taille initiale : {df.shape}")

    # Supprimer doublons et remplir les valeurs manquantes
    df = df.drop_duplicates().fillna("")

    # Nettoyage du texte
    if "text" in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)
    elif "content" in df.columns:
        df["clean_text"] = df["content"].apply(clean_text)
    else:
        print(f"Aucune colonne 'text' ou 'content' dans {file_name}, saut du fichier.")
        continue

    # --- Découpage en chunks avec chevauchement ---
    chunks = []
    for _, row in df.iterrows():
        for chunk in chunk_text_overlap(row["clean_text"], max_words=100, overlap=20):
            chunks.append({
                "title": row.get("title", ""),    
                "subject": row.get("subject", ""), 
                "date": row.get("date", ""),      
                "chunk": chunk
            })
    
    chunks_df = pd.DataFrame(chunks)

    # --- Sauvegarde ---
    clean_file_name = f"chunked_{file_name}"
    clean_file_path = os.path.join(PROCESSED_PATH, clean_file_name)
    chunks_df.to_csv(clean_file_path, index=False)

    print(f"{len(chunks_df)} chunks créés et enregistrés dans {clean_file_path}")
