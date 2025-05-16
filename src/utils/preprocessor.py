from typing import Union, Optional
import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer


class DocumentProcessor:
    def __init__(self, methods: Optional[Union[str, list[str], tuple[str, ...]]] = None):
        """
        Инициализация процессора документов.
        
        Args:
            methods: None, одно название метода или последовательность названий методов.
        """
        if methods is None:
            self.methods: list[str] = []
        elif isinstance(methods, str):
            self.methods = [methods]
        elif isinstance(methods, (list, tuple)):
            self.methods = list(methods)
        else:
            raise ValueError("methods должен быть None, str, list или tuple")

        self.stemmer = SnowballStemmer("russian")
        # Убедимся, что русские стоп-слова доступны
        try:
            self.stopwords_ru = stopwords.words("russian")
        except LookupError:
            nltk.download('stopwords')
            self.stopwords_ru = stopwords.words("russian")
        self.lemmatizer = MorphAnalyzer()

    def get_methods(self):
        """Возвращает список методов предобработки."""
        return self.methods

    def process(self, text):
        """
        Применяет все выбранные методы предобработки к тексту.
        
        Args:
            text: Исходный текст для обработки.
            
        Returns:
            Обработанный текст.
        """
        for method in self.methods:
            try:
                text = self.processing_methods[method](self, text)
            except KeyError:
                print('Такой метод обработки не существует')
                raise
        return text
    
    def normalize_spaces(self, text):
        """Нормализует пробелы в тексте."""
        assert isinstance(text, str)
        return ' '.join(text.split())
    
    def lowcase_process(self, text):
        """Приводит текст к нижнему регистру."""
        assert isinstance(text, str)
        return text.lower()
    
    def special_chars(self, text):
        """Удаляет специальные символы из текста."""
        assert isinstance(text, str)
        return re.sub('[^A-Za-z0-9А-Яа-я ]+', '', text)
    
    def remove_stopwords(self, text):
        """Удаляет стоп-слова из текста."""
        preprocessed_tokens = [x for x in text.split() if x not in self.stopwords_ru]
        return ' '.join(preprocessed_tokens)
    
    def lemmatize_text(self, text):
        """Лемматизирует текст."""
        tokens = text.split()
        lemmatized_tokens = [self.lemmatizer.parse(token)[0].normal_form for token in tokens]
        return ' '.join(lemmatized_tokens)

    processing_methods = {
        'normalize_spaces': normalize_spaces,
        'lowcase': lowcase_process,
        'special_chars': special_chars,
        'remove_stopwords': remove_stopwords,
        'lemmatize_text': lemmatize_text
    }
    