from article_processor import ArticleProcessor
import re
import spacy

class PreprocessingArticle(ArticleProcessor):
    def __init__(self, text, label=None, lemmatize=True, remove_stopwords=True, nlp_model=None):
        super().__init__(text, label)
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.nlp = nlp_model 

    def clean_text(self):
        text = self.text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        self.text = text
        return self.text

    def lemmatize_text(self):
        if self.lemmatize and self.nlp:
            doc = self.nlp(self.text)
            self.text = " ".join([token.lemma_ for token in doc if not token.is_stop or not self.remove_stopwords])
        return self.text
