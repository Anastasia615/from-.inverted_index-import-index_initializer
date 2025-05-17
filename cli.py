# cli.py
import click
import sqlite3

from src.index_creater.inverted_index import index_initializer
from src.utils.preprocessor import DocumentProcessor


@click.group()
@click.option('--database-path', '-d', required=True,
              help='Путь к файлу SQLite (таблица ParsedData.publication_content)')
@click.option('--methods', '-m', multiple=True,
              default=['lowcase', 'normalize_spaces', 'special_chars', 'remove_stopwords'],
              help='Методы предобработки текста')
@click.option('--encoding', '-e', type=click.Choice(['gamma', 'delta']), default=None,
              help='Алгоритм сжатия: gamma или delta')
@click.pass_context
def cli(ctx, database_path, methods, encoding):
    """
    Консольное приложение для работы с инвертированным индексом.
    """
    preprocessor = DocumentProcessor(list(methods))
    ctx.obj = {
        'index': index_initializer(database_path, preprocessor, encoding)
    }


@cli.command()
@click.argument('query', nargs=-1)
@click.pass_context
def search(ctx, query):
    """
    Поиск документов по запросу.
    
    Args:
        query: Поисковый запрос (одно или несколько слов).
    """
    q = ' '.join(query)
    idx = ctx.obj['index']
    doc_ids = idx.search(q)
    
    if not doc_ids:
        click.echo('Ничего не найдено.')
        return
    
    # Получаем тексты документов из базы данных
    db_path = ctx.parent.params['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    click.echo(f'Найдено {len(doc_ids)} документов:')
    
    # Выводим первые 5 результатов
    for doc_id in doc_ids[:5]:
        cursor.execute('SELECT publication_content FROM ParsedData WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        if result:
            # Сокращаем текст для вывода
            text = result[0]
            short_text = text[:100] + '...' if len(text) > 100 else text
            click.echo(f'{doc_id}: {short_text}')
    
    conn.close()


if __name__ == '__main__':
    cli() 