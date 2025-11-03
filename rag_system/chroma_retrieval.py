# database/chroma_utils.py
import os
from openai import AzureOpenAI
from database.chroma_utils import get_embedding


# Initialisation du client Azure OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embedding(text: str):
    """
    Génère l'embedding du texte fourni à l'aide du modèle Azure OpenAI.
    """
    if not text or not text.strip():
        return None

    try:
        response = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERREUR EMBEDDING] {e}")
        return None
