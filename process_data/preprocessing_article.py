import re
from bs4 import BeautifulSoup
import unicodedata
import spacy
from process_data.article_processor import ArticleProcessor

class PreprocessingArticle(ArticleProcessor):
    def __init__(self, text, label=None, lemmatize=True, nlp_model=None, preserve_entities=True):
        super().__init__(text, label)
        self.lemmatize = lemmatize
        self.nlp = nlp_model
        self.preserve_entities = preserve_entities

    def clean_text(self):
        text = self.text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()  
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)   
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')  
        text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)     
        text = re.sub(r"\s+", " ", text).strip()             
        self.text = text
        return self.text

    def lemmatize_text(self):
        if not self.lemmatize or not self.nlp:
            return self.text

        doc = self.nlp(self.text)
        processed_tokens = []

        for token in doc:
            if self.preserve_entities and token.ent_type_:
                processed_tokens.append(token.text)  
            else:
                processed_tokens.append(token.lemma_) 

        self.text = " ".join(processed_tokens)
        return self.text