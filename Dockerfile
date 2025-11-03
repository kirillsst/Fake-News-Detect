# Exemple minimal pour Streamlit + Azure OpenAI
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

COPY .env .env

# CMD avec variable Azure $PORT
CMD streamlit run interface/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
