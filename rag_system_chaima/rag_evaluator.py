# rag_system_chaima/rag_evaluator.py

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .rag_pipeline import rag_pipeline
from .chroma_retrieval import get_collection

CSV_FILE = "rag_system_chaima/eval_examples.csv"

def evaluate_rag_pipeline():
    # Charger les exemples
    df = pd.read_csv(CSV_FILE)

    y_true = []
    y_pred = []

    print("=== Évaluation RAG sur échantillon CSV ===")

    for idx, row in df.iterrows():
        text = row["text"]
        label = row["label"]

        print(f"\nTexte : {text[:80]}...")  # Affiche un extrait
        print(f"Label réel : {label}")

        # Appel du pipeline RAG
        verdict_str = rag_pipeline(text)  # retourne "Verdict: TRUE\nExplanation: ..."
        
        # Extraire le verdict
        verdict_line = [line for line in verdict_str.splitlines() if line.upper().startswith("VERDICT:")]
        verdict = verdict_line[0].split(":", 1)[1].strip() if verdict_line else "UNKNOWN"

        print(f"Verdict RAG : {verdict}")

        y_true.append(label.lower())
        y_pred.append(verdict.lower())

    # Calcul des métriques
    print("\n=== Résultats métriques ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score (macro):", f1_score(y_true, y_pred, average="macro"))
    print("\nRapport complet :\n", classification_report(y_true, y_pred, digits=3))

if __name__ == "__main__":
    evaluate_rag_pipeline()
