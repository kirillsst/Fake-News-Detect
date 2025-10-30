"""
Script d'évaluation du système RAG pour la détection de fake news.

Ce script permet de :
- Charger un échantillon de textes depuis le CSV de chunks
- Exécuter le pipeline RAG sur chaque texte
- Comparer les prédictions avec les labels réels (ground truth)
- Calculer les métriques de performance (accuracy, precision, recall, F1)
- Sauvegarder les résultats détaillés dans un fichier CSV
"""

import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rag_system.rag_pipeline import rag_analyze, get_label_from_csv 

#  CONFIGURATION 
CSV_PATH = "data/processed/chunks.csv"  
SAMPLE_SIZE = 200  # Nombre d'échantillons à tester
SAVE_PATH = "evaluation_results.csv"  # Fichier de sortie pour les résultats détaillés
MAX_CHUNKS = 20  # Nombre maximum de chunks à récupérer 
MAX_CHARS = 1000  # Nombre maximum de caractères par chunk


def normalize_label(l):
    """
    Normalise les labels pour assurer la cohérence.
    
    Convertit les labels en format standardisé (TRUE/FAKE) et gère les cas particuliers.
    
    Args:
        l: Label brut (peut être str, None, ou autre type)
        
    Returns:
        str ou None: "TRUE", "FAKE", ou None si le label est invalide
        
    Exemples:
        - "false" ou "FALSE" → "FAKE"
        - "true" ou "TRUE" → "TRUE"
        - Valeurs invalides → None
    """
    if l is None:
        return None
    
    # Conversion en string et nettoyage
    s = str(l).strip().upper()
    
    # Correction du label FALSE en FAKE pour cohérence
    if s == "FALSE":
        s = "FAKE"
    
    # Validation : seuls TRUE et FAKE sont acceptés
    if s not in ["TRUE", "FAKE"]:
        return None
    
    return s


def evaluate():
    """
    Fonction principale d'évaluation du système RAG.
    
    Processus :
    1. Charge les données depuis le CSV
    2. Échantillonne aléatoirement (si SAMPLE_SIZE est défini)
    3. Exécute rag_analyze sur chaque texte
    4. Compare prédictions vs ground truth
    5. Calcule et affiche les métriques de performance
    6. Sauvegarde les résultats détaillés
    
    Returns:
        tuple: (DataFrame des résultats, tuple des métriques (acc, prec, rec, f1, cm))
    """
   
    df = pd.read_csv(CSV_PATH)
    
    # Échantillonnage aléatoire pour tests rapides
    if SAMPLE_SIZE:
        df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    
    records = []  # Liste pour stocker les résultats de chaque prédiction
    t0 = time.time()  # Début du chronomètre global
    
    #  BOUCLE D'ÉVALUATION 
    for idx, row in df.iterrows():
        text = row["text"]
        
        # Exécution du pipeline RAG et mesure du temps
        start = time.time()
        res = rag_analyze(text, max_chunks=MAX_CHUNKS, max_chars=MAX_CHARS)
        duration = time.time() - start
        
        # Récupération du label réel (ground truth) avec fallback robuste
        # Essaie d'abord get_label_from_csv, puis le champ "label" du CSV
        gt = normalize_label(get_label_from_csv(text) or row.get("label"))
        
        # Extraction de la prédiction du modèle
        pred = normalize_label(res.get("verdict"))
        
        # NOTE: Calcul de la précision de retrieval (retrieval_precision_at_k)
        # Pour calculer cette métrique, il faudrait que rag_analyze retourne
        # les labels individuels de chaque chunk récupéré (context_metadatas).
        # Actuellement non implémenté → valeur None
        retrieval_precision_at_k = None
        
        # Enregistrement des résultats pour cette prédiction
        records.append({
            "index": idx,
            "text_snippet": text[:200].replace("\n", " "),  # Extrait du texte pour référence
            "ground_truth": gt,
            "prediction": pred,
            "correct": (gt is not None and pred == gt),  # Prédiction correcte ?
            "rag_duration": duration,  # Temps d'exécution en secondes
            "result_text": res.get("result_text")  # Réponse complète du RAG
        })
        
        # Affichage de la progression tous les 10 échantillons
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(df)}] traités - dernier temps : {duration:.2f}s")
    
    #  SAUVEGARDE DES RÉSULTATS DÉTAILLÉS 
    results_df = pd.DataFrame(records)
    results_df.to_csv(SAVE_PATH, index=False)
    print(f"\nRésultats détaillés sauvegardés → {SAVE_PATH}")
    
    #  CALCUL DES MÉTRIQUES 
    # Filtrer uniquement les lignes avec un ground truth valide
    eval_df = results_df[results_df["ground_truth"].notnull()].copy()
    y_true = eval_df["ground_truth"]
    y_pred = eval_df["prediction"].fillna("UNKNOWN")  # Remplacer None par "UNKNOWN"
    
    # Ne garder que les prédictions valides (TRUE ou FAKE)
    mask = y_pred.isin(["TRUE", "FAKE"])
    y_true2 = y_true[mask]
    y_pred2 = y_pred[mask]
    
    # Vérification : au moins une prédiction valide
    if len(y_pred2) == 0:
        print("Aucune prédiction valide à évaluer (toutes UNKNOWN).")
        return
    
    # Calcul des métriques de classification
    acc = accuracy_score(y_true2, y_pred2)  # Taux de prédictions correctes
    prec = precision_score(y_true2, y_pred2, pos_label="TRUE")  # Précision pour la classe TRUE
    rec = recall_score(y_true2, y_pred2, pos_label="TRUE")  # Rappel pour la classe TRUE
    f1 = f1_score(y_true2, y_pred2, pos_label="TRUE")  # Score F1 pour la classe TRUE
    cm = confusion_matrix(y_true2, y_pred2, labels=["TRUE", "FAKE"])  # Matrice de confusion
    
    #  AFFICHAGE DU RÉSUMÉ 
    print("\n=== RÉSUMÉ DE L'ÉVALUATION ===")
    print(f"Échantillons évalués (avec ground truth & prédiction valide) : {len(y_pred2)}")
    print(f"Accuracy (exactitude globale) : {acc:.4f}")
    print(f"Precision (TRUE) : {prec:.4f}")
    print(f"Recall (TRUE) : {rec:.4f}")
    print(f"F1-Score (TRUE) : {f1:.4f}")
    print("\nMatrice de confusion (lignes=vrais labels, colonnes=prédictions)")
    print("                TRUE  FAKE")
    print(f"TRUE (réel)     {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"FAKE (réel)     {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    return results_df, (acc, prec, rec, f1, cm)


#  POINT D'ENTRÉE 
if __name__ == "__main__":
   
    evaluate()