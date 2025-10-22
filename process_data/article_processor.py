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
