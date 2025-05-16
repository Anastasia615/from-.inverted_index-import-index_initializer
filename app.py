import argparse
from flask import Flask, request 
from src.index_creater import index_initializer
from src.utils import DocumentProcessor


app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_route():
    """Проверка работоспособности приложения."""
    return "App is working"

@app.route('/documents', methods=['GET'])
def document_search_route():
    """
    Поиск документов по запросу.
    
    Returns:
        Список найденных документов.
    """
    query = request.args.get('query')
    if not query:
        return {"error": "Query parameter is required"}, 400
        
    normalized_query = preprocessor.process(query)
    result = inverted_index.search(normalized_query)
    
    return [x[1] for x in result]


@app.route('/indexes', methods=['GET'])
def indices_search_route():
    """
    Поиск индексов документов по запросу.
    
    Returns:
        Список индексов найденных документов.
    """
    query = request.args.get('query')
    if not query:
        return {"error": "Query parameter is required"}, 400
        
    normalized_query = preprocessor.process(query)
    result = inverted_index.search(normalized_query)
    return [x[0] for x in result]


def main():
    """Основная функция запуска приложения."""
    parser = argparse.ArgumentParser(description='HTTP API для поиска по инвертированному индексу')
    
    parser.add_argument('-p', '--database_path', 
                        help='Путь к существующей базе данных', 
                        required=True)
    
    parser.add_argument('-m', '--methods', 
                        help='Методы предобработки документов',
                        required=False,
                        action='store',
                        dest='methods',
                        nargs='*',
                        default=['lowcase', 'normalize_spaces', 'special_chars',
                                'remove_stopwords', 'lemmatize_text'])

    parser.add_argument('-e', '--encoding',
                        help='Метод сжатия индекса для уменьшения размера',
                        required=False,
                        choices=['delta', 'gamma'])

    args = parser.parse_args()

    global preprocessor, inverted_index
    preprocessor = DocumentProcessor(methods=args.methods)
    inverted_index = index_initializer(args.database_path, preprocessor=preprocessor, encoding=args.encoding)

    app.run()


if __name__ == '__main__':
    main()


