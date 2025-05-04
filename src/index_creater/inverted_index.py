"""
src/index_creater/inverted_index.py

Improved inverted index implementation with optional compression,
type hints, and simplified postings building.
"""

from collections import defaultdict
import os
import pickle
import hashlib
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from src.utils.bm_search import bm_search
from src.utils.db_reader import read_whole_content
from src.utils.encoder import EncodedInvertedIndex


class InvertedIndex:
    """
    Simple inverted index with optional Elias encoding for compression.
    """
    def __init__(self,
                 documents: List[str],
                 preprocessor,
                 load_path: Optional[str] = None,
                 encoding: Optional[str] = None
                 ) -> None:
        self.documents = documents
        self.preprocessor = preprocessor
        self.encoding = encoding
        self.index: dict[str, np.ndarray] = {}

        if load_path and os.path.exists(load_path):
            self._load_index(load_path)
        else:
            self._build_index()
            if self.encoding:
                self.index = EncodedInvertedIndex(self.index,
                                                  encoding_method=self.encoding)

    def _build_index(self) -> None:
        """
        Build the inverted index from the list of documents.
        """
        postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, doc in enumerate(self.documents):
            text = self.preprocessor.process(doc)
            for word in text.split():
                postings[word].append(doc_id)
        # convert postings to sorted unique numpy arrays
        self.index = {
            word: np.array(sorted(set(ids)), dtype=np.int32)
            for word, ids in postings.items()
        }

    def _load_index(self, path: str) -> None:
        """
        Load a stored inverted index from disk.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if self.encoding:
            idx = EncodedInvertedIndex({}, encoding_method=self.encoding)
            idx.load_encoded_dict(data)
            self.index = idx
        else:
            self.index = data

    def save_index(self, path: str) -> None:
        """
        Save the inverted index (encoded or raw) to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            if self.encoding:
                pickle.dump(self.index.get_encoded_dict(), f)
            else:
                pickle.dump(self.index, f)

    def _intersect_postings(self, words: List[str]) -> List[int]:
        """
        Return sorted list of document IDs containing all the given words.
        """
        postings = [self.index.get(w, np.array([], dtype=np.int32))
                    for w in words]
        if not postings:
            return []
        result = postings[0]
        for p in postings[1:]:
            result = np.intersect1d(result, p)
        return result.tolist()

    def search(self, query: str) -> List[Tuple[int, str]]:
        """
        Search for documents matching the query.
        Returns list of (doc_id, original_document).
        """
        processed = self.preprocessor.process(query)
        words = processed.split()
        doc_ids = self._intersect_postings(words)
        results: List[Tuple[int, str]] = []
        for doc_id in doc_ids:
            text = self.preprocessor.process(self.documents[doc_id])
            if bm_search(text, query):
                results.append((doc_id, self.documents[doc_id]))
        return results


def _get_index_path(db_path: str,
                    methods: List[str],
                    encoding: Optional[str]
                    ) -> str:
    """
    Generate a unique file path for the index based on DB path,
    preprocessing methods, and encoding.
    """
    key = db_path + str(encoding) + '-' + '-'.join(methods)
    filename = hashlib.md5(key.encode()).hexdigest() + '.pickle'
    index_dir = os.path.join(os.path.dirname(db_path), 'index')
    return os.path.join(index_dir, filename)


def index_initializer(database_path: str,
                      preprocessor,
                      encoding: Optional[str] = None
                      ) -> InvertedIndex:
    """
    Initialize the inverted index: load from disk if exists,
    or build and save.
    """
    methods = preprocessor.get_methods()
    index_path = _get_index_path(database_path, methods, encoding)
    # ensure index directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    load_path = index_path if os.path.exists(index_path) else None
    documents = read_whole_content(database_path)
    idx = InvertedIndex(documents, preprocessor,
                        load_path=load_path, encoding=encoding)
    if load_path is None:
        idx.save_index(index_path)
    return idx







        


    

