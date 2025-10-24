# tests/test_pipeline.py
import pytest
import pandas as pd
from process_data.pipeline import ArticlePipeline
import os

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un CSV temporaire pour tester le pipeline."""
    csv_content = """title,text,subject,date
Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing,"Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that...",News,December 31, 2017
Drunk Bragging Trump Staffer Started Russian Collusion Investigation,"House Intelligence Committee Chairman Devin Nunes is going to have a bad day...",News,December 31, 2017
"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file_path = raw_dir / "sample.csv"
    file_path.write_text(csv_content)
    return raw_dir

@pytest.fixture
def processed_dir(tmp_path):
    """Dossier temporaire pour sauvegarder le CSV processed."""
    processed = tmp_path / "processed"
    processed.mkdir()
    return processed

def test_article_pipeline_process_all(sample_csv, processed_dir):
    """Teste le pipeline complet : lecture CSV, nettoyage, chunks, sauvegarde."""
    pipeline = ArticlePipeline(
        raw_path=str(sample_csv),
        processed_path=str(processed_dir),
        chunk_size=5,
        overlap=1,
        lemmatize=False  # Pour accélérer le test
    )
    
    chunks = pipeline.process_all()

    # Vérifie que des chunks ont été créés
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Vérifie que chaque chunk a les colonnes attendues
    for chunk in chunks:
        assert "article_id" in chunk
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "label" in chunk

    # Vérifie que le CSV final a été créé
    output_file = processed_dir / "chunks.csv"
    assert output_file.exists()

    # Vérifie le contenu du CSV
    df = pd.read_csv(output_file)
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) == len(chunks)