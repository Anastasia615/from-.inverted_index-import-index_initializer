from typing import Union, Sequence, Optional
import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer


class DocumentProcessor:
    def __init__(self, methods: Optional[Union[str, Sequence[str]]] = None):
        """
        methods: None, a single method name or a sequence of method names.
        """
        if methods is None:
            self.methods: list[str] = []
        elif isinstance(methods, str):
            self.methods = [methods]
        elif isinstance(methods, (list, tuple)):
            self.methods = list(methods)
        else:
            raise ValueError("methods must be None, str, list or tuple")

        self.stemmer = SnowballStemmer("russian")
        # Ensure Russian stopwords are available
        try:
            self.stopwords_ru = stopwords.words("russian")
        except LookupError:
            nltk.download('stopwords')
            self.stopwords_ru = stopwords.words("russian")
        self.lemmatizer = MorphAnalyzer()


    def get_methods(self):
        return self.methods

    def process(self, text):
        for method in self.methods:
            try:
                text = self.processing_methods[method](self, text)
            except KeyError:
                print('No such processing method')
                raise
        return text
    
    def normalize_spaces(self, text):
        assert isinstance(text, str)
        return ' '.join(text.split())
    
    def lowcase_process(self, text):
        assert isinstance(text, str)
        return text.lower()
    
    def special_chars(self, text):
        assert isinstance(text, str)
        return re.sub('[^A-Za-z0-9А-Яа-я ]+', '', text)
    
    def remove_stopwords(self, text):
        preprocessed_tokens = [x for x in text.split() if x not in self.stopwords_ru]
        return ' '.join(preprocessed_tokens)


    def lemmatize_text(self, text):
        preprocessed_tokens = [self.lemmatizer.normal_forms(x)[0] for x in text.split()]  
        return ' '.join(preprocessed_tokens)

    

    
    processing_methods = {
        'normalize_spaces':normalize_spaces,
        'lowcase':lowcase_process,
        'special_chars':special_chars,
        'remove_stopwords':remove_stopwords,
        'lemmatize_text':lemmatize_text
    }
    