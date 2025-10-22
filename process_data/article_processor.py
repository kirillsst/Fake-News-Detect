class ArticleProcessor:
    def __init__(self, text, label=None, article_id=None):
        self.text = text
        self.label = label
        self.article_id = article_id

    def get_text(self):
        return self.text

    def get_label(self):
        return self.label

    def get_article_id(self):
        return self.article_id
