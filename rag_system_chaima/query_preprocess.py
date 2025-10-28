# query_preprocess.py : Prétraitement du texte utilisateur

import re

def preprocess_query(text: str) -> str:
    """
    Prétraitement simple :
    - Minuscules
    - Suppression des espaces multiples
    - Suppression des URL et caractères spéciaux
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # supprimer URL
    text = re.sub(r'\s+', ' ', text).strip()  # espaces multiples
    text = re.sub(r'[^\w\s]', '', text)  # caractères spéciaux
    return text
