import unittest
import tempfile
import time
from pathlib import Path
from src.index_creater.inverted_index import InvertedIndex
from src.utils.preprocessor import DocumentProcessor

class TestInvertedIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc_processor = DocumentProcessor(methods=[
            'lowcase', 
            'normalize_spaces', 
            'special_chars', 
            'remove_stopwords', 
            'lemmatize_text'
        ])
        
        # Генерация тестовых документов
        cls.documents = {
            1: "Ректор СПбГУ объявил о новых инициативах",
            2: "В МГУ прошла конференция по информационному поиску",
            3: "СПбГУ и МГУ вошли в топ-100 мирового рейтинга",
        }

    def test_index_creation(self):
        index = InvertedIndex(":memory:", self.doc_processor)
        
        # Добавление документов
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        
        # Проверка наличия терминов
        expected_terms = ['ректор', 'спбгу', 'мгу', 'конференция', 'топ', '100']
        for term in expected_terms:
            self.assertIn(term, index.index)

    def test_search_functionality(self):
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        
        # Тест простого поиска
        results = index.search("ректор спбгу")
        self.assertEqual(results, [1, 3])

        # Тест поиска с отсутствующим термином
        results = index.search("несуществующий термин")
        self.assertEqual(results, [])

class TestIndexPerformance(unittest.TestCase):
    def test_indexing_speed(self):
        doc_processor = DocumentProcessor()
        index = InvertedIndex(":memory:", doc_processor)
        
        # Генерация 1000 тестовых документов
        test_docs = {i: f"Document {i} about universities" for i in range(1000)}
        
        start_time = time.time()
        for doc_id, text in test_docs.items():
            index.add_document(doc_id, text)
        elapsed = time.time() - start_time
        
        print(f"\nИндексирование 1000 документов: {elapsed:.2f} сек")
        self.assertLess(elapsed, 2.0)

if __name__ == '__main__':
    unittest.main()
