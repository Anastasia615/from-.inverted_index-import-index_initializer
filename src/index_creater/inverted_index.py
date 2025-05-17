"""
src/index_creater/inverted_index.py

Improved inverted index implementation with optional compression,
type hints, and simplified postings building.
"""

from collections import defaultdict
from pathlib import Path
import pickle
import hashlib
from typing import Optional, Dict
import json

import numpy as np
from tqdm import tqdm

from src.utils.bm_search import bm_search
from src.utils.db_reader import read_whole_content
from src.utils.encoder import EncodedInvertedIndex, AbstractEncoder, EliasGammaEncoder, EliasDeltaEncoder
from src.utils.preprocessor import DocumentProcessor


class InvertedIndex:
    """
    Simple inverted index with optional Elias encoding for compression.
    """
    def __init__(self, index_path: str, preprocessor: DocumentProcessor, encoder: Optional[AbstractEncoder] = None):
        """
        Инициализация инвертированного индекса
        
        Args:
            index_path: Путь к директории с индексом
            preprocessor: Объект для предобработки текста
            encoder: Опциональный кодировщик для сжатия постинг-листов
        """
        self.index_path = Path(index_path)
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.index: Dict[str, list[int]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Загрузка индекса из файла"""
        index_file = self.index_path / 'index.pkl'
        if index_file.exists():
            with open(index_file, 'rb') as f:
                self.index = pickle.load(f)

    def save_index(self) -> None:
        """Сохранение индекса в файл"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path / 'index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

    def add_document(self, doc_id: int, text: str) -> None:
        """
        Добавление документа в индекс
        
        Args:
            doc_id: ID документа
            text: Текст документа
        """
        # Предобработка текста
        processed_text = self.preprocessor.process(text)
        
        # Добавление термов в индекс
        for term in processed_text.split():
            if term not in self.index:
                self.index[term] = []
            if doc_id not in self.index[term]:
                self.index[term].append(doc_id)
                self.index[term].sort()

    def _intersect_postings(self, postings1: list[int], postings2: list[int]) -> list[int]:
        """
        Пересечение двух постинг-листов
        
        Args:
            postings1: Первый постинг-лист
            postings2: Второй постинг-лист
            
        Returns:
            Список общих document ID
        """
        result = []
        i, j = 0, 0
        
        while i < len(postings1) and j < len(postings2):
            if postings1[i] == postings2[j]:
                result.append(postings1[i])
                i += 1
                j += 1
            elif postings1[i] < postings2[j]:
                i += 1
            else:
                j += 1
                
        return result

    def search(self, query: str) -> list[int]:
        """
        Поиск документов по запросу
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список ID документов, содержащих все термы запроса
        """
        # Предобработка запроса
        processed_query = self.preprocessor.process(query)
        terms = processed_query.split()
        
        if not terms:
            return []
            
        # Получение постинг-листов для каждого терма
        postings = []
        for term in terms:
            if term in self.index:
                postings.append(self.index[term])
            else:
                return []  # Если хотя бы один терм не найден, возвращаем пустой результат
                
        # Пересечение постинг-листов
        result = postings[0]
        for posting in postings[1:]:
            result = self._intersect_postings(result, posting)
            
        return result


def _get_index_path(db_path: str,
                    methods: list[str],
                    encoding: Optional[str]
                    ) -> str:
    """
    Генерация уникального пути к файлу индекса на основе пути к БД,
    методов предобработки и типа кодирования.
    
    Args:
        db_path: Путь к базе данных
        methods: Список методов предобработки
        encoding: Тип кодирования (сжатия)
        
    Returns:
        Строка с путем к файлу индекса
    """
    key = db_path + str(encoding) + '-' + '-'.join(methods)
    filename = hashlib.md5(key.encode()).hexdigest() + '.pickle'
    index_dir = Path(db_path).parent / 'index'
    return str(index_dir / filename)


def index_initializer(database_path: str,
                      preprocessor,
                      encoding: Optional[str] = None
                      ) -> InvertedIndex:
    """
    Инициализация инвертированного индекса: загрузка с диска, если существует,
    или создание и сохранение нового.
    
    Args:
        database_path: Путь к базе данных
        preprocessor: Объект для предобработки текста
        encoding: Тип кодирования (сжатия)
        
    Returns:
        Объект инвертированного индекса
    """
    methods = preprocessor.get_methods()
    
    # Вместо создания директории под каждый индекс, используем общую директорию
    index_path = 'data/index'
    print(f"Используется индексный путь: {index_path}")
    
    # Создаем экземпляр индекса
    idx = InvertedIndex(index_path, preprocessor, encoding)
    
    return idx







        


    

