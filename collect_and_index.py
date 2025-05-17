import sqlite3
from datetime import datetime
from pathlib import Path
from src.index_creater.inverted_index import index_initializer
from src.utils.preprocessor import DocumentProcessor
from src.index_creater.inverted_index import InvertedIndex
from src.utils.db_reader import read_whole_content
import csv

def create_database():
    """Создает SQLite базу данных с необходимой структурой."""
    conn = sqlite3.connect('data/telegram_data.sqlite')
    cursor = conn.cursor()
    
    # Создаем таблицу для хранения данных
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ParsedData (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        university TEXT,
        publication_date TIMESTAMP,
        publication_content TEXT,
        views INTEGER,
        forwards INTEGER
    )
    ''')
    
    conn.commit()
    return conn

def save_to_database(conn, data):
    """Сохраняет собранные данные в базу данных."""
    cursor = conn.cursor()
    
    for item in data:
        cursor.execute('''
        INSERT INTO ParsedData (university, publication_date, publication_content, views, forwards)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            item['university'],
            item['publication_date'],
            item['message'],
            item['views'],
            item['forwards']
        ))
    
    conn.commit()

def import_csv_data(csv_path):
    """Импортирует данные из telegram_stats.csv в формате, подходящем для базы."""
    data = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'university': row.get('university', ''),
                'publication_date': row.get('publication_date', ''),
                'message': row.get('message', ''),
                'views': int(row.get('views', 0)),
                'forwards': int(row.get('forwards', 0)),
            })
    return data

def main():
    # Создаем директорию для данных, если её нет
    Path('data').mkdir(exist_ok=True)
    
    # Создаем базу данных
    conn = create_database()
    
    try:
        # Импортируем данные из CSV
        print("Импортируем данные из telegram_stats.csv...")
        telegram_data = import_csv_data('telegram_stats.csv')
        
        # Сохраняем данные в базу
        print("Сохраняем данные в базу...")
        save_to_database(conn, telegram_data)
        
        # Создаем инвертированный индекс
        print("Создаем инвертированный индекс...")
        preprocessor = DocumentProcessor([
            'lowcase',
            'normalize_spaces',
            'special_chars',
            'remove_stopwords'
        ])
        
        index_path = 'data/index'
        idx = InvertedIndex(index_path, preprocessor)
        
        # Если индекс не существует (не загружен), читаем документы и сохраняем индекс
        index_file = Path(index_path) / 'index.pkl'
        if not index_file.exists():
            print("Индекс не найден, создаём новый...")
            documents = read_whole_content('data/telegram_data.sqlite')
            print(f"Прочитано {len(documents)} документов из базы данных")
            
            # Добавляем документы в индекс
            print("Добавляем документы в индекс...")
            cursor = conn.cursor()
            cursor.execute('SELECT id, publication_content FROM ParsedData')
            for doc_id, content in cursor.fetchall():
                idx.add_document(doc_id, content)
                
            # Сохраняем индекс
            print("Сохраняем индекс...")
            idx.save_index()
            
        print(f"Собрано и проиндексировано {len(telegram_data)} документов")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main() 