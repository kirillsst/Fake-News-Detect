# query_preprocess.py : Prétraitement de la requête utilisateur
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.preprocessing_article import PreprocessingArticle

def preprocess_query(text: str, lemmatize: bool = True) -> str:
    """
    Nettoie et (facultativement) lemmatise le texte utilisateur.
    """
    article = PreprocessingArticle(text, lemmatize=lemmatize)
    clean_text = article.clean_text()
    if lemmatize:
        clean_text = article.lemmatize_text()
    return clean_text
