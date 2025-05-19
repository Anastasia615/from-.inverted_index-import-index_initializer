import unittest
import tempfile
import time
from pathlib import Path
from src.index_creater.inverted_index import InvertedIndex
from src.utils.preprocessor import DocumentProcessor

class TestInvertedIndex(unittest.TestCase):
    """
    Тесты для проверки корректности работы инвертированного индекса и поиска.
    """
    @classmethod
    def setUpClass(cls):
        # Создание процессора документов с набором методов предобработки
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
        """
        Проверяет, что после добавления документов в индекс
        все ожидаемые термины присутствуют в структуре индекса.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        expected_terms = ['ректор', 'спбгу', 'мгу', 'конференция', 'топ', '100']
        for term in expected_terms:
            self.assertIn(term, index.index)

    def test_search_functionality(self):
        """
        Проверяет корректность поиска по нескольким терминам и отсутствие результатов по несуществующему термину.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("ректор спбгу")
        self.assertEqual(results, [1, 3])
        results = index.search("несуществующий термин")
        self.assertEqual(results, [])

    def test_case_insensitivity(self):
        """
        Проверяет, что поиск не зависит от регистра букв.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("СПБГУ")
        self.assertEqual(results, [1, 3])

    def test_special_characters(self):
        """
        Проверяет, что специальные символы в запросе не мешают поиску.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("топ-100!!!")
        self.assertEqual(results, [3])

    def test_stopwords_removal(self):
        """
        Проверяет, что стоп-слова корректно удаляются и не влияют на поиск.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("и о по")
        self.assertEqual(results, [])

    def test_lemmatization(self):
        """
        Проверяет, что поиск работает по леммам (например, "инициатива" найдёт "инициативах").
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("инициатива")
        self.assertEqual(results, [1])

    def test_empty_query(self):
        """
        Проверяет, что пустой запрос возвращает пустой результат.
        """
        index = InvertedIndex(":memory:", self.doc_processor)
        for doc_id, text in self.documents.items():
            index.add_document(doc_id, text)
        results = index.search("")
        self.assertEqual(results, [])

class TestIndexPerformance(unittest.TestCase):
    """
    Тест производительности индексирования большого количества документов.
    """
    def test_indexing_speed(self):
        """
        Проверяет, что индексирование 1000 документов происходит достаточно быстро.
        """
        doc_processor = DocumentProcessor()
        index = InvertedIndex(":memory:", doc_processor)
        test_docs = {i: f"Document {i} about universities" for i in range(1000)}
        start_time = time.time()
        for doc_id, text in test_docs.items():
            index.add_document(doc_id, text)
        elapsed = time.time() - start_time
        print(f"\nИндексирование 1000 документов: {elapsed:.2f} сек")
        self.assertLess(elapsed, 2.0)

if __name__ == '__main__':
    unittest.main()
