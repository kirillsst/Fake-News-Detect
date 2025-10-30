import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

CSV_SAMPLE_PATH = "data/sample_texts.csv"
CSV_RESULTS_PATH = "data/rag_results_metrics.csv"

def evaluate_dataset(csv_path=CSV_SAMPLE_PATH, save_path=CSV_RESULTS_PATH):
    df = pd.read_csv(csv_path)
    results = []
    y_true = []
    y_pred = []

    from rag_system.rag_core import rag_analyze  # import sûr

    for i, row in df.iterrows():
        text = row["text"]
        label = row["label"].upper()
        print(f"[{i+1}/{len(df)}] Analyse du texte ({label}) ...")

        res = rag_analyze(text)
        verdict_line = [line for line in res["result_text"].splitlines() if line.strip().upper().startswith("VERDICT:")]
        verdict = verdict_line[0].split(":", 1)[1].strip().upper() if verdict_line else "FAKE"
        if verdict not in ["TRUE", "FAKE"]:
            verdict = "FAKE"

        results.append({
            "text": text,
            "label": label,
            "verdict": verdict,
            "num_chunks": res["num_chunks"],
            "eval_duration": res["eval_duration"]
        })

        y_true.append(label)
        y_pred.append(verdict)

    # Precision / Recall / F1
    precision_true = precision_score(y_true, y_pred, pos_label="TRUE", zero_division=0)
    recall_true = recall_score(y_true, y_pred, pos_label="TRUE", zero_division=0)
    f1_true = f1_score(y_true, y_pred, pos_label="TRUE", zero_division=0)

    precision_fake = precision_score(y_true, y_pred, pos_label="FAKE", zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label="FAKE", zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label="FAKE", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=["TRUE", "FAKE"])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0

    metrics_df = pd.DataFrame(results)
    metrics_df["precision_true"] = precision_true
    metrics_df["recall_true"] = recall_true
    metrics_df["f1_true"] = f1_true
    metrics_df["precision_fake"] = precision_fake
    metrics_df["recall_fake"] = recall_fake
    metrics_df["f1_fake"] = f1_fake
    metrics_df["accuracy"] = accuracy

    metrics_df.to_csv(save_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {save_path}")
    print(f"Accuracy global: {accuracy:.3f}, Precision TRUE: {precision_true:.3f}, Recall TRUE: {recall_true:.3f}")
    print(f"Precision FAKE: {precision_fake:.3f}, Recall FAKE: {recall_fake:.3f}")

if __name__ == "__main__":
    evaluate_dataset()
