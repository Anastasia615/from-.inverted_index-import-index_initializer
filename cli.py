# cli.py
import click

from src.index_creater.inverted_index import index_initializer
from src.utils.preprocessor import DocumentProcessor


@click.group()
@click.option('--database-path', '-d', required=True,
              help='Путь к файлу SQLite (таблица ParsedData.publication_content)')
@click.option('--methods', '-m', multiple=True,
              default=['lowcase', 'normalize_spaces', 'special_chars', 'remove_stopwords', 'lemmatize_text'],
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
    """
    q = ' '.join(query)
    idx = ctx.obj['index']
    results = idx.search(q)
    if not results:
        click.echo('Ничего не найдено.')
        return
    for doc_id, text in results:
        click.echo(f'{doc_id}: {text}')


if __name__ == '__main__':
    cli() 