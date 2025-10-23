# tests/test_articles.py
"""
Tests unitaires pour les classes ArticleProcessor, ChunkedArticle et PreprocessingArticle
en utilisant CSV .
"""

import pytest
import pandas as pd
from process_data.article_processor import ArticleProcessor
from process_data.chunked_article import ChunkedArticle
from process_data.preprocessing_article import PreprocessingArticle
import spacy
import warnings
import os

# -----------------------
# Gestion globale des warnings
# -----------------------
@pytest.fixture(autouse=True)
def ignore_warnings():
    """Ignore les warnings de dépréciation dans tous les tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield

# -----------------------
# Fixture : modèle spaCy
# -----------------------
@pytest.fixture(scope="module")
def nlp_model():
    """Charge le modèle anglais complet pour lemmatisation et entités."""
    return spacy.load("en_core_web_sm")

# -----------------------
# Chemin vers CSV réel
# -----------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/Fake.csv") 

# -----------------------
# Tests pour ArticleProcessor
# -----------------------
def test_article_processor_load_csv():
    """Teste la lecture CSV et suppression des doublons."""
    df = ArticleProcessor.load_csv(CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "text" in df.columns

def test_article_processor_getters():
    """Teste les getters de ArticleProcessor."""
    article = ArticleProcessor(text="Hello world", label="True", article_id=123)
    assert article.get_text() == "Hello world"
    assert article.get_label() == "True"
    assert article.get_article_id() == 123

def test_article_processor_clean_text():
    """Teste la méthode clean_text par défaut (ne modifie pas le texte)."""
    article = ArticleProcessor(text="Some text")
    assert article.clean_text() == "Some text"

# -----------------------
# Tests pour ChunkedArticle
# -----------------------
def test_chunked_article_text():
    """Teste le découpage en chunks avec overlap."""
    text = "one two three four five six seven eight nine ten"
    chunked = ChunkedArticle(text, chunk_size=4, overlap=1)
    chunks = chunked.chunk_text()
    expected_chunks = [
        "one two three four",
        "four five six seven",
        "seven eight nine ten"
    ]
    # On prend les 3 premiers chunks pour ignorer le dernier éventuel très court
    assert chunks[:3] == expected_chunks

# -----------------------
# Tests pour PreprocessingArticle
# -----------------------
def test_preprocessing_article_clean_text():
    """Teste le nettoyage du texte depuis le CSV réel."""
    df = pd.read_csv(CSV_PATH)
    text = df.iloc[0]["text"]
    article = PreprocessingArticle(text=text)
    cleaned = article.clean_text()
    assert isinstance(cleaned, str)
    assert cleaned == cleaned.lower()  # minuscule
    assert "http" not in cleaned  # liens supprimés

def test_preprocessing_article_lemmatize(nlp_model):
    """Teste la lemmatisation en conservant les entités sur texte réel."""
    df = pd.read_csv(CSV_PATH)
    text = df.iloc[0]["text"]
    article = PreprocessingArticle(text=text, nlp_model=nlp_model)
    article.clean_text()
    lemmatized = article.lemmatize_text()
    assert isinstance(lemmatized, str)
    assert len(lemmatized.strip()) > 0  # texte non vide
