import sqlite3


class SQLite():
    """
    Контекстный менеджер для работы с SQLite базой данных.
    
    Обеспечивает корректное подключение и закрытие соединения с базой данных.
    """
    def __init__(self, file='sqlite.db'):
        """
        Инициализация менеджера SQLite.
        
        Args:
            file: Путь к файлу базы данных.
        """
        self.file=file
    def __enter__(self):
        """
        Подключение к базе данных при входе в контекст.
        
        Returns:
            Курсор для выполнения SQL-запросов.
        """
        self.conn = sqlite3.connect(self.file)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()
    def __exit__(self, type, value, traceback):
        """
        Закрытие соединения при выходе из контекста.
        """
        self.conn.commit()
        self.conn.close()

def read_whole_content(path, table_name='ParsedData', content_column = 'publication_content'):
    """
    Чтение всего контента из указанной таблицы базы данных.
    
    Args:
        path: Путь к файлу базы данных.
        table_name: Имя таблицы (по умолчанию 'ParsedData').
        content_column: Имя столбца с контентом (по умолчанию 'publication_content').
        
    Returns:
        Список строк текста из указанного столбца.
    """
    with SQLite(path) as cur:
        content = cur.execute(f'SELECT {content_column} FROM {table_name};').fetchall()

    normalized_list = [x[0] for x in content]

    return normalized_list




