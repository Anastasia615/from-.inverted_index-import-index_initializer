import os
import sqlite3
from datetime import datetime
from data_collector import collect_telegram_data
from src.index_creater.inverted_index import index_initializer
from src.utils.preprocessor import DocumentProcessor

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
    os.makedirs('data', exist_ok=True)
    
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
        
        index = index_initializer(
            database_path='data/telegram_data.sqlite',
            preprocessor=preprocessor,
            encoding='delta'  # или 'gamma'
        )
        
        print(f"Собрано и проиндексировано {len(telegram_data)} документов")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main() 