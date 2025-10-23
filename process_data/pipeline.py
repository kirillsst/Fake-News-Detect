import os
import pandas as pd
import spacy
from preprocessing_article import PreprocessingArticle
from chunked_article import ChunkedArticle

class ArticlePipeline:
    """
    Pipeline complet de prétraitement des articles :
    1. Lecture des fichiers CSV
    2. Ajout d'une colonne label
    3. Nettoyage du texte
    4. Lemmatisation (optionnelle)
    5. Découpage en chunks
    6. Sauvegarde dans un CSV processed
    """
    def __init__(self, raw_path, processed_path, chunk_size=200, overlap=50,
                 lemmatize=True):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lemmatize = lemmatize

        # Charger le modèle spaCy une seule fois
        if self.lemmatize:
            print("Chargement du modèle spaCy...")
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None

    def process_all(self):
        csv_files = [f for f in os.listdir(self.raw_path) if f.endswith(".csv")]
        all_chunks = []

        for file_name in csv_files:
            file_path = os.path.join(self.raw_path, file_name)
            print(f"\nTraitement du fichier : {file_name}")

            df = pd.read_csv(file_path)
            df = df.drop_duplicates().fillna("")

            # Ajouter la colonne label
            if "fake" in file_name.lower():
                df["label"] = "Fake"
            else:
                df["label"] = "True"

            for idx, row in df.iterrows():
                # Nettoyage du texte + lemmatisation
                article = PreprocessingArticle(
                    text=row['text'],
                    label=row['label'],
                    lemmatize=self.lemmatize,
                    nlp_model=self.nlp
                )
                clean_text = article.clean_text()
                if self.lemmatize:
                    clean_text = article.lemmatize_text()

                # Découpage en chunks
                chunked = ChunkedArticle(
                    text=clean_text,
                    label=row['label'],
                    chunk_size=self.chunk_size,
                    overlap=self.overlap
                )
                chunks = chunked.chunk_text()

                # Sauvegarder tous les chunks
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "article_id": idx,
                        "chunk_id": i,
                        "text": chunk,
                        "label": row['label']
                    })

        # Sauvegarder le tout dans un seul CSV
        os.makedirs(self.processed_path, exist_ok=True)
        output_file = os.path.join(self.processed_path, "chunks.csv")
        pd.DataFrame(all_chunks).to_csv(output_file, index=False)
        print(f"\nLes chunks traités sont sauvegardés dans {output_file}")
        return all_chunks

if __name__ == "__main__":
    RAW_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"

    pipeline = ArticlePipeline(
        raw_path=RAW_PATH,
        processed_path=PROCESSED_PATH,
        chunk_size=200,
        overlap=50,
        lemmatize=True,  
    )
    pipeline.process_all()
