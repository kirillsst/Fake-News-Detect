import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from rag_system.rag_pipeline import rag_analyze

# Chemin vers ton CSV d'échantillon
CSV_SAMPLE_PATH = "data/sample_texts.csv"
CSV_RESULTS_PATH = "data/rag_results_metrics.csv"

def main():
    df = pd.read_csv(CSV_SAMPLE_PATH)

    results = []
    y_true = []
    y_pred = []

    for i, row in df.iterrows():
        text = row["text"]
        label = row["label"].upper()
        print(f"[{i+1}/{len(df)}] Analyse du texte ({label}) ...")
        
        # Limiter max_chunks et max_chars pour accélérer les tests
        res = rag_analyze(text, max_chunks=10, max_chars=1000)  # 10 chunks au lieu de 3
        
        verdict = res["verdict"]
        # Transformer UNKNOWN en FAKE pour métriques (optionnel)
        if verdict not in ["TRUE", "FAKE"]:
            verdict = "FAKE"

        results.append({
            "text": text,
            "label": label,
            "verdict": verdict
        })

        y_true.append(label)
        y_pred.append(verdict)

    # Calcul des métriques avec average='binary' pour TRUE
    precision = precision_score(y_true, y_pred, pos_label="TRUE")
    recall = recall_score(y_true, y_pred, pos_label="TRUE")
    f1 = f1_score(y_true, y_pred, pos_label="TRUE")

    # Calcul des métriques pour FAKE aussi
    precision_fake = precision_score(y_true, y_pred, pos_label="FAKE", zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label="FAKE", zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label="FAKE", zero_division=0)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=["TRUE", "FAKE"])
    tn, fp, fn, tp = cm.ravel()

    print("\n=== Métriques RAG ===")
    print("\nMatrice de confusion:")
    print(f"                 Prédit TRUE  |  Prédit FAKE")
    print(f"Réel TRUE     |     {tn:3d}      |     {fp:3d}")
    print(f"Réel FAKE     |     {fn:3d}      |     {tp:3d}")

    print("\nMétriques pour TRUE:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")

    print("\nMétriques pour FAKE (classe à détecter):")
    print(f"  Precision: {precision_fake:.3f}")
    print(f"  Recall:    {recall_fake:.3f}")
    print(f"  F1-score:  {f1_fake:.3f}")

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    print(f"\nAccuracy: {accuracy:.3f}")

    # Sauvegarde - TOUTES les colonnes AVANT d'écrire le CSV
    metrics_df = pd.DataFrame(results)
    metrics_df["precision"] = precision
    metrics_df["recall"] = recall
    metrics_df["f1_score"] = f1
    metrics_df["precision_fake"] = precision_fake  
    metrics_df["recall_fake"] = recall_fake        
    metrics_df["f1_fake"] = f1_fake                
    metrics_df["accuracy"] = accuracy              
    
    metrics_df.to_csv(CSV_RESULTS_PATH, index=False)
    print(f" Résultats sauvegardés dans {CSV_RESULTS_PATH}")

if __name__ == "__main__":
    main()