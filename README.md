# 📘 Guide de configuration et d’exécution du projet *Fake News Detect*

## ⚙️ 0. Installation des dépendances

Il faut créer l'environnement 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Avant de commencer, assurez-vous d’avoir installé et synchronisé toutes les dépendances du projet à l’aide de **uv** :

```bash
uv sync
```

Télécharger le modèle

```bash
python -m spacy download en_core_web_sm
```

## 1. Chargement des données

Utilisez le lien suivant pour télécharger le jeu de données :  
👉 [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data)

Après le téléchargement, placez les fichiers dans le répertoire racine du projet :
___

```bash
/Fake-News-Detect/
│
├── True.csv
└── Fake.csv
```

Créez ensuite une structure de dossiers pour organiser vos données :

___

```bash
/Fake-News-Detect/
│
└── data/
├── raw/
└── processed/
```

Pour extraire un nombre spécifique de lignes d’un fichier CSV (par exemple, pour un test rapide), utilisez la commande suivante :

```bash
head -n <n_lignes> input.csv > data/raw/output.csv
```
Remplacez <n_lignes> par le nombre de lignes souhaité.

## 2. Exécution du pipeline de traitement
Lancez le script suivant pour traiter vos deux fichiers CSV (True.csv et Fake.csv) :

```bash
python process_data/pipeline.py
```

Une fois le traitement terminé, un fichier CSV final sera généré dans le dossier :

```bash
data/processed/
```

Ce fichier fusionnera les deux sources et contiendra les colonnes suivantes :

| Colonnes         | Description                              |
| ---------------- | -----------------------------------------|
| article_id       | Identifiant unique de l’article          |
| chunk_id         | Identifiant du segment (chunk) de texte  |
| text             | Contenu textuel de l’article             |
| label            | Classe de l’article : Fake ou True       |

## 3. Création de la base de données Chroma
Tout se fait désormais en une seule commande via le pipeline principal :
```bash
python database/chroma_pipeline.py
```
Ce script :

  - lit automatiquement le fichier data/processed/chunks.csv,

  - crée la collection fake_news_collection (si elle n’existe pas),

  - génère les embeddings pour chaque chunk,

  - insère les données dans la base par lots (batch_size=50 par défaut).

Après l’exécution, vérifiez le contenu de la base de données avec :
```bash
python database/check_db.py
```
Ce script vous permettra de confirmer le nombre de chunks présents dans la collection.

## 4. Utilisation de l’interface Streamlit
Une fois la base configurée, vous pouvez lancer l’interface utilisateur avec Streamlit :
```bash
streamlit run interface/app.py
```

Cette interface vous permettra d’entrer un texte et d’obtenir une prédiction au format suivant :
```bash
Résultat : Fake ou True
Explication : Raisonnement du modèle
```
## Structure de projet

```bash
├── README.md
├── chroma_db
├── data
│   ├── processed
│   │   └── chunks.csv
│   └── raw
│       ├── Fake.csv
│       └── True.csv
├── database
│   ├── __init__.py
│   ├── check_db.py
│   ├── chroma_insert.py
│   ├── chroma_pipeline.py
│   ├── chroma_setup.py
│   └── chroma_utils.py
├── interface
│   └── app.py
├── notebooks
│   └── exploration.ipynb
├── process_data 
│   ├── article_processor.py
│   ├── base_article.py
│   ├── chunked_article.py
│   ├── pipeline.py
│   └── preprocessing_article.py
├── pyproject.toml
├── rag_system
│   ├── __init__.py
│   ├── chroma_retrieval.py
│   ├── ollama_generation.py
│   ├── query_preprocess.py
│   └── rag_pipeline.py
├── requirements.txt
└── uv.lock
```
