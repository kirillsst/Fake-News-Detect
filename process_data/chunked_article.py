from article_processor import ArticleProcessor

class ChunkedArticle(ArticleProcessor):
    def __init__(self, text, label=None, chunk_size=200, overlap=50):
        super().__init__(text, label)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, max_words=None, overlap=None):
        max_words = max_words or self.chunk_size
        overlap = overlap or self.overlap
        words = self.text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunks.append(" ".join(words[start:end]))
            start += max_words - overlap
        return chunks

