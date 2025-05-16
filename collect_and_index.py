import sqlite3
from datetime import datetime
from pathlib import Path
from data_collector import collect_telegram_data
from src.index_creater.inverted_index import index_initializer
from src.utils.preprocessor import DocumentProcessor
from src.index_creater.inverted_index import InvertedIndex
from src.utils.file_utils import read_whole_content

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

def main():
    # Создаем директорию для данных, если её нет
    Path('data').mkdir(exist_ok=True)
    
    # Создаем базу данных
    conn = create_database()
    
    try:
        # Собираем данные из Telegram
        print("Собираем данные из Telegram...")
        telegram_data = collect_telegram_data()
        
        # Сохраняем данные в базу
        print("Сохраняем данные в базу...")
        save_to_database(conn, telegram_data)
        
        # Создаем инвертированный индекс
        print("Создаем инвертированный индекс...")
        preprocessor = DocumentProcessor([
            'lowcase',
            'normalize_spaces',
            'special_chars',
            'remove_stopwords',
            'lemmatize_text'
        ])
        
        index_path = 'data/telegram_data.sqlite'
        idx = InvertedIndex(index_path, preprocessor, encoding='delta')
        
        # Если индекс не существует (не загружен), читаем документы и сохраняем индекс
        index_file = Path(index_path) / 'index.pkl'
        if not index_file.exists():
            documents = read_whole_content(index_path)
            idx.save_index()
        
        print(f"Собрано и проиндексировано {len(telegram_data)} документов")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main() 