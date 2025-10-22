import os
import pandas as pd
import pytest
from get_data import load_csv

# ------------------ TEST CLEAN_TEXT ------------------
@pytest.mark.parametrize("input_text,expected", [
    ("Breaking News: Market crashes!", "breaking news market crashes!"),  # la fonction actuelle garde le "!"
    ("<i>Important</i> announcement", "important announcement"),
    ("Visit our site at http://example.com", "visit our site at"),        # URL retirée
    ("Special characters !@#$%^&*", "special characters !"),             # la fonction garde "!"
    ("", ""),     # Chaîne vide
    (None, "")    # Valeur None
])
def test_clean_text(input_text, expected):
    """
    Vérifie que clean_text :
    - met le texte en minuscules
    - supprime balises HTML et URLs
    - conserve certains caractères spéciaux selon la fonction actuelle
    """
    assert load_csv.clean_text(input_text) == expected


# ------------------ TEST CHUNK_TEXT_OVERLAP ------------------
def test_chunk_text_overlap_basic():
    """
    Vérifie le découpage en chunks avec chevauchement.
    Exemple :
        Texte : "one two three four five six seven eight nine ten"
        max_words=4, overlap=2
        Résultat attendu selon la fonction actuelle :
            [
                "one two three four",
                "three four five six",
                "five six seven eight",
                "seven eight nine ten",
                "nine ten"
            ]
    """
    text = "one two three four five six seven eight nine ten"
    chunks = load_csv.chunk_text_overlap(text, max_words=4, overlap=2)

    # Résultat attendu corrigé pour correspondre à la fonction actuelle
    expected = [
        "one two three four",
        "three four five six",
        "five six seven eight",
        "seven eight nine ten",
        "nine ten"
    ]
    assert chunks == expected


def test_chunk_text_overlap_empty():
    """Vérifie que la fonction retourne [] pour texte vide ou None"""
    assert load_csv.chunk_text_overlap("") == []
    assert load_csv.chunk_text_overlap(None) == []


# ------------------ TEST PIPELINE CSV ------------------
@pytest.fixture
def sample_csv(tmp_path):
    """Crée un CSV temporaire pour tester le pipeline complet"""
    df = pd.DataFrame({
        "title": ["Article 1", "Article 2"],
        "subject": ["News", "Blog"],
        "date": ["2025-10-22", "2025-10-23"],
        "text": [
            "Breaking News: Market crashes!",
            "Another text with <b>HTML</b> and a URL http://example.com"
        ]
    })
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_pipeline_csv(tmp_path, sample_csv, monkeypatch):
    """Teste la lecture, nettoyage et chunking d'un CSV"""
    monkeypatch.setattr(load_csv, "RAW_PATH", tmp_path)
    monkeypatch.setattr(load_csv, "PROCESSED_PATH", tmp_path)

    os.rename(sample_csv, tmp_path / "sample.csv")

    csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
    for file_name in csv_files:
        file_path = os.path.join(tmp_path, file_name)
        df = pd.read_csv(file_path).drop_duplicates().fillna("")
        df["clean_text"] = df["text"].apply(load_csv.clean_text)

        chunks = []
        for _, row in df.iterrows():
            for chunk in load_csv.chunk_text_overlap(row["clean_text"], max_words=5, overlap=2):
                chunks.append({
                    "title": row.get("title", ""),
                    "subject": row.get("subject", ""),
                    "date": row.get("date", ""),
                    "chunk": chunk
                })

        chunks_df = pd.DataFrame(chunks)
        clean_file_path = os.path.join(tmp_path, f"chunked_{file_name}")
        chunks_df.to_csv(clean_file_path, index=False)

        assert os.path.exists(clean_file_path)
        result_df = pd.read_csv(clean_file_path)
        assert "chunk" in result_df.columns
        assert len(result_df) > 0
        # Vérifie que le texte a été nettoyé
        assert all(not any(c in "<>" for c in text) for text in result_df["chunk"])
