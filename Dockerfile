# Base Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers nécessaires d'abord pour le cache pip
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt


COPY .env .env

# CMD avec variable Azure $PORT

# Copier tout le projet
COPY . .

# Assurer que le dossier ChromaDB existe et a les bonnes permissions
RUN mkdir -p /app/chroma_db && chmod -R 777 /app/chroma_db

# Lancer Streamlit avec la variable PORT Azure

CMD streamlit run interface/app.py --server.port $PORT --server.address 0.0.0.0
