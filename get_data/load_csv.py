import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import nltk
import unicodedata

nltk.download('punkt', quiet=True)

# --- Définition des chemins ---
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Fonction de nettoyage du texte ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Supprimer HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Supprimer URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Normaliser Unicode en ASCII (convertit “ ’ – … en ' - etc.)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Supprimer tout ce qui n’est pas une lettre ou un espace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Supprimer espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tout mettre en minuscules
    return text.lower()


# --- Fonction de découpage (chunking) avec chevauchement ---
def chunk_text_overlap(text, max_words=4, overlap=20):
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

# --- Liste tous les fichiers CSV dans le dossier RAW ---
csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
print("Fichiers détectés :", csv_files)

for file_name in csv_files:
    file_path = os.path.join(RAW_PATH, file_name)
    print(f"\n--- Traitement du fichier : {file_name} ---")

    # Lecture du fichier CSV
    df = pd.read_csv(file_path)
    print(f"Taille initiale : {df.shape}")
    print("Colonnes :", df.columns.tolist())

    # Supprimer doublons et remplir les valeurs manquantes
    df = df.drop_duplicates().fillna("")

    # Détection automatique de la colonne texte
    text_col = None
    for candidate in ["text", "content", "body", "article"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if not text_col:
        print(f"Aucune colonne de texte trouvée dans {file_name}, fichier ignoré.")
        continue

    # Nettoyage du texte
    df["clean_text"] = df[text_col].apply(clean_text)

    # Colonne "valid_text" = True si texte non vide
    df["valid_text"] = df["clean_text"].apply(lambda x: bool(x.strip()))

    # --- Découpage du texte en chunks ---
    chunks = []
    for _, row in df.iterrows():
        if not row["valid_text"]:
            continue  # Ignore les lignes vides
        for chunk in chunk_text_overlap(row["clean_text"], max_words=50, overlap=20):
            chunks.append({
                "title": row.get("title", ""),
                "subject": row.get("subject", ""),
                "date": row.get("date", ""),
                "chunk": chunk
            })

    chunks_df = pd.DataFrame(chunks)

    # --- Sauvegarde du fichier chunké ---
    clean_file_path = os.path.join(PROCESSED_PATH, file_name)
    chunks_df.to_csv(clean_file_path, index=False)
    print(f"{len(chunks_df)} chunks enregistrés dans {clean_file_path}")

    # --- Sauvegarde du fichier de statuts True/False ---
    status_file = f"status_{file_name}"
    status_path = os.path.join(PROCESSED_PATH, status_file)
    df[["title", text_col, "clean_text", "valid_text"]].to_csv(status_path, index=False)
    print(f"Fichier des statuts créé : {status_path}")

    # --- Aperçu rapide ---
    print("\nAperçu du fichier des statuts :")
    print(df[["clean_text", "valid_text"]].head(5))
