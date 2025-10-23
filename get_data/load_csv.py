import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import unicodedata

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Supprimer HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Supprimer URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Normaliser Unicode
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Supprimer tout sauf lettres et espaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Supprimer espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Minuscules
    return text.lower().strip()

def chunk_text_overlap(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks

# --- Traitement CSV ---
csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(RAW_PATH, file_name)
    df = pd.read_csv(file_path, encoding='utf-8')  # encodage explicite
    
    # Détection texte
    text_col = None
    for c in ["text", "content", "body", "article"]:
        if c in df.columns:
            text_col = c
            break
    if not text_col:
        continue
    
    # Nettoyage complet
    df['clean_text'] = df[text_col].apply(clean_text)
    
    # Création chunks
    chunks = []
    for _, row in df.iterrows():
        if not row['clean_text']:
            continue
        for ch in chunk_text_overlap(row['clean_text'], max_words=100, overlap=20):
            chunks.append({
                "title": row.get("title", ""),
                "subject": row.get("subject", ""),
                "date": row.get("date", ""),
                "chunk": ch
            })
    
    chunks_df = pd.DataFrame(chunks)
    
    # Sauvegarde
    chunks_df.to_csv(os.path.join(PROCESSED_PATH, f"chunked_{file_name}"), index=False)
    print(f"{len(chunks_df)} chunks sauvegardés pour {file_name}")
