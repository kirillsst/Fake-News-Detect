import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk
import logging

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# load NLTK et Punkt je l'utilise pour la tokenisation des phrases
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

RAW_PATH = "data/raw"         
PROCESSED_PATH = "data/processed"  

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)



def clean_text(text):
    """
    Nettoie un texte de manière complète :
    - Supprime les balises HTML
    - Retire les URLs
    - Supprime les caractères spéciaux
    - Réduit les espaces multiples
    - Convertit tout en minuscules
    """
    if not isinstance(text, str):
        return ""

    # Suppression des balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Suppression des URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Suppression des caractères spéciaux, en gardant la ponctuation de base
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)

    # Réduction des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    # Conversion en minuscules
    text = text.lower()
    return text


def chunk_text(text, max_sentences=5):
    """
    Découpe un texte en sous-parties (chunks) de maximum 'max_sentences' phrases.
    Chaque chunk contiendra entre 1 et 5 phrases selon la taille du texte.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    # Tokenisation du texte en phrases
    sentences = sent_tokenize(text)

    # Regroupement des phrases en blocs de taille fixe
    return [" ".join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]




# Liste tous les fichiers CSV présents dans le dossier "data/raw"
csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]

# Boucle sur chaque fichier pour le traiter
for file_name in csv_files:
    file_path = os.path.join(RAW_PATH, file_name)
    logging.info(f"Traitement du fichier : {file_name}")

    # Chargement du fichier CSV
    df = pd.read_csv(file_path)
    logging.info(f"Taille initiale : {df.shape}")

    # Suppression des doublons et remplissage des valeurs manquantes
    df = df.drop_duplicates().fillna("")

    # Nettoyage du texte selon la colonne disponible
    if "text" in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)
    elif "content" in df.columns:
        df["clean_text"] = df["content"].apply(clean_text)
    else:
        # Si aucune colonne pertinente n’est trouvée, on ignore ce fichier 
        logging.warning(f"Aucune colonne 'text' ou 'content' dans {file_name}, saut du fichier.")
        continue

    # --- Découpage du texte en chunks ---
    chunks = []
    for _, row in df.iterrows():
        for chunk in chunk_text(row["clean_text"]):
            chunks.append({
                "title": row.get("title", ""),    
                "subject": row.get("subject", ""), 
                "date": row.get("date", ""),      
                "chunk": chunk                   
            })

    chunks_df = pd.DataFrame(chunks)

    # --- Sauvegarde du fichier traité ---
    clean_file_name = f"chunked_{file_name}"
    clean_file_path = os.path.join(PROCESSED_PATH, clean_file_name)
    chunks_df.to_csv(clean_file_path, index=False)

    logging.info(f"{len(chunks_df)} chunks créés et enregistrés dans {clean_file_path}")
