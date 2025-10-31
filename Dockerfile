# Exemple minimal pour Streamlit
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# CMD avec variable Azure $PORT
CMD streamlit run interface/app.py --server.port $PORT --server.address 0.0.0.0
