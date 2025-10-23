from abc import ABC, abstractmethod

class BaseArticle(ABC):
    """
    Classe abstraite de base pour tout type d'article.
    Définit l'interface minimale que chaque sous-classe doit implémenter.
    """

    def __init__(self, text, label=None):
        self.text = text
        self.label = label

    @abstractmethod
    def load_csv(file_path):
        """Charge un CSV et retourne un DataFrame."""
        pass

    @abstractmethod
    def clean_text(self):
        """Nettoie le texte brut et retourne la version nettoyée."""
        pass

    @abstractmethod
    def get_text(self):
        """Retourne le texte de l'article."""
        pass

    @abstractmethod
    def get_label(self):
        """Retourne le label de l'article."""
        pass
