"""
Пакет utils содержит вспомогательные модули для работы с данными и индексами:
- Чтение данных из базы данных
- Предобработка текста
- Алгоритмы поиска
- Кодирование и сжатие индексов
"""

from .db_reader import read_whole_content
from .bm_search import bm_search
from .preprocessor import DocumentProcessor
from .encoder import EncodedInvertedIndex, EliasDeltaEncoder, EliasGammaEncoder

__all__ = ['read_whole_content', 'bm_search', 'DocumentProcessor']

