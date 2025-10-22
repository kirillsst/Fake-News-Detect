from process_data.article_processor import ArticleProcessor

class ChunkedArticle(ArticleProcessor):
    def clean_text(self):
        print("Nettoyage avancé du texte...")
        # Implémentation réelle du nettoyage

    def split_chunks(self, chunk_size=200, overlap=50):
        print(f"Découpage du texte en chunks de {chunk_size} mots avec overlap {overlap}")
        # Implémentation réelle du découpage
        # self.chunks = [...]

    def generate_embeddings(self):
        print("Génération des embeddings via Ollama")
        # Implémentation réelle de la génération d'embeddings
        # self.embeddings = [...]
