import argparse
from flask import Flask, request 
from src.index_creater import index_initializer
from src.utils import DocumentProcessor



app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_route():
    return "App is working"

@app.route('/documents', methods=['GET'])
def document_search_route():
    query = request.args.get('query')
    normalized_query = preprocessor.process(query)
    result = inverted_index.search(normalized_query)
    
    return [x[1] for x in result]


@app.route('/indexes', methods=['GET'])
def indices_search_route():
    query = request.args.get('query')
    normalized_query = preprocessor.process(query)
    result = inverted_index.search(normalized_query)
    return [x[0] for x in result]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p','--database_path', 
                        help='Path to existing database', 
                        required=True)
    
    parser.add_argument('-m', '--methods', 
                        help='Methods of preprocessing documents',
                        required=False,
                        action='store',
                        dest='methods',
                        nargs='*',
                        default=['lowcase', 'normalize_spaces', 'special_chars',
                                 'remove_stopwords','lemmatize_text'])

    parser.add_argument('-e', '--encoding',
                        help='Does index needs to be encoded to reduce its size',
                        required=False,
                        choices=['delta', 'gamma'])

    args = parser.parse_args()

    preprocessor = DocumentProcessor(methods=args.methods)
    inverted_index = index_initializer(args.database_path, preprocessor=preprocessor, encoding=args.encoding)

    app.run()


