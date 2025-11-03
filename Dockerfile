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
CMD streamlit run interface/app.py --server.port $PORT --server.address 0.0.0.0
