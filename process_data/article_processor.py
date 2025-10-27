import pandas as pd
# from base_article import BaseArticle
from .base_article import BaseArticle


class ArticleProcessor(BaseArticle):
    """
    Classe concrète qui implémente la logique de base définie dans BaseArticle.
    Sert de fondation pour les classes plus spécialisées (prétraitement, découpage, etc.)
    """

    def __init__(self, text, label=None, article_id=None):
        super().__init__(text, label)
        self.article_id = article_id

    @staticmethod
    def load_csv(file_path):import pandas as pd
from process_data.base_article import BaseArticle

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
