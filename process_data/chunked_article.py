from article_processor import ArticleProcessor

class ChunkedArticle(ArticleProcessor):
    def __init__(self, text, label=None, chunk_size=200, overlap=50):
        super().__init__(text, label)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self):
        words = self.text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i+self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
