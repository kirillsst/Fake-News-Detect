from abc import ABC, abstractmethod

class ArticleProcessor(ABC):
    def __init__(self, text: str, label: str = None):
        self.text = text
        self.label = label
        self.chunks = []
        self.embeddings = []

    @abstractmethod
    def clean_text(self):
        """
        Nettoyage de base : minuscules, suppression HTML, URLs, caractères spéciaux
        Doit être implémenté dans la classe enfant
        """
        pass

    @abstractmethod
    def split_chunks(self, chunk_size: int = 200, overlap: int = 50):
        """
        Découpe le texte en chunks avec chevauchement
        Doit être implémenté dans la classe enfant
        """
        pass

    @abstractmethod
    def generate_embeddings(self):
        """
        Génère embeddings via Ollama pour tous les chunks
        Doit être implémenté dans la classe enfant
        """
        pass
import pandas as pd
from base_article import BaseArticle

class ArticleProcessor(BaseArticle):
    """
    Classe concrète qui implémente la logique de base définie dans BaseArticle.
    Sert de fondation pour les classes plus spécialisées (prétraitement, découpage, etc.)
    """

    def __init__(self, text, label=None, article_id=None):
        super().__init__(text, label)
        self.article_id = article_id

    @staticmethod
    def load_csv(file_path):
        """Charge un CSV et retourne un DataFrame nettoyé."""
        df = pd.read_csv(file_path)
        df = df.drop_duplicates().fillna("")
        print(f"CSV chargé : {file_path} ({len(df)} lignes)")
        return df

    def clean_text(self):
        """Méthode de base : ne fait rien, à redéfinir dans les sous-classes."""
        return self.text

    def get_text(self):
        return self.text

    def get_label(self):
        return self.label

    def get_article_id(self):
        return self.article_id

from abc import ABC, abstractmethod

class ArticleProcessor(ABC):
    def __init__(self, text: str, label: str = None):
        self.text = text
        self.label = label
        self.chunks = []
        self.embeddings = []

    @abstractmethod
    def clean_text(self):
        """
        Nettoyage de base : minuscules, suppression HTML, URLs, caractères spéciaux
        Doit être implémenté dans la classe enfant
        """
        pass

    @abstractmethod
    def split_chunks(self, chunk_size: int = 200, overlap: int = 50):
        """
        Découpe le texte en chunks avec chevauchement
        Doit être implémenté dans la classe enfant
        """
        pass

    @abstractmethod
    def generate_embeddings(self):
        """
        Génère embeddings via Ollama pour tous les chunks
        Doit être implémenté dans la classe enfant
        """
        pass
