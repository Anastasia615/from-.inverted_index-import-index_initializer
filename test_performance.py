import unittest
import time
import os
from pathlib import Path
from src.index_creater.inverted_index import InvertedIndex, index_initializer
from src.utils.preprocessor import DocumentProcessor

class TestIndexingPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = Path("data/test_performance")
        cls.test_data_path.mkdir(exist_ok=True)
        
        # Генерация тестового набора данных (40k документов)
        cls.generate_test_docs(40000)

    @staticmethod
    def generate_test_docs(num_docs):
        docs = {}
        for i in range(num_docs):
            text = f"Документ {i} о деятельности ректора СПбГУ" if i % 2 == 0 \
                else f"Документ {i} о научных достижениях МГУ"
            docs[i] = text
        return docs

    def test_compression_impact(self):
        # Конфигурации тестирования
        configs = [
            {'encoding': None, 'name': 'Без сжатия'},
            {'encoding': 'gamma', 'name': 'Gamma-сжатие'},
            {'encoding': 'delta', 'name': 'Delta-сжатие'}
        ]
        
        results = {}
        docs = self.generate_test_docs(40000)
        
        for config in configs:
            processor = DocumentProcessor()
            index = index_initializer(
                database_path=str(self.test_data_path / "test_db.sqlite"),
                preprocessor=processor,
                encoding=config['encoding']
            )
            
            # Замер времени индексирования
            start_time = time.time()
            for doc_id, text in docs.items():
                index.add_document(doc_id, text)
            index.save_index()
            elapsed = time.time() - start_time
            
            # Замер размера индекса
            index_size = os.path.getsize(self.test_data_path / "index" / "index.pkl")
            
            results[config['name']] = {
                'time': elapsed,
                'size': index_size
            }
        
        # Вывод результатов
        print("\nРезультаты тестирования сжатия:")
        for name, data in results.items():
            print(f"{name}:")
            print(f"  Время индексирования: {data['time']:.2f} сек")
            print(f"  Размер индекса: {data['size'] / 1024 / 1024:.2f} МБ")

    def test_search_performance(self):
        docs = self.generate_test_docs(40000)
        processor = DocumentProcessor()
        index = index_initializer(
            database_path=str(self.test_data_path / "test_db.sqlite"),
            preprocessor=processor
        )
        
        # Добавление документов в индекс
        for doc_id, text in docs.items():
            index.add_document(doc_id, text)
        
        # Тестирование скорости поиска
        test_cases = [
            "ректор спбгу",
            "мгу научный достижение",
            "несуществующий запрос"
        ]
        
        results = {}
        for query in test_cases:
            start_time = time.time()
            results = index.search(query)
            elapsed = time.time() - start_time
            print(f"\nПоиск: '{query}'")
            print(f"  Найдено документов: {len(results)}")
            print(f"  Время выполнения: {elapsed:.4f} сек")
            self.assertLess(elapsed, 0.5)

if __name__ == '__main__':
    unittest.main()
