import sqlite3


class SQLite():
    def __init__(self, file='sqlite.db'):
        self.file=file
    def __enter__(self):
        self.conn = sqlite3.connect(self.file)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()
    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()

def read_whole_content(path, table_name='ParsedData', content_column = 'publication_content'):
    """
    Function reads all publication_content from database and return list of text
    """
    with SQLite(path) as cur:
        content = cur.execute(f'SELECT {content_column} FROM {table_name};').fetchall()

    normalized_list = [x[0] for x in content]

    return normalized_list




