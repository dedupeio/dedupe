import Levenshtein_search

from .index import Index
from .core import Enumerator


class LevenshteinIndex(Index):
    def __init__(self):
        self.index_key = Levenshtein_search.populate_wordset(-1, [])
        self._doc_to_id = Enumerator(start=1)

    def index(self, doc):
        if doc not in self._doc_to_id:
            self._doc_to_id[doc]
            Levenshtein_search.add_string(self.index_key, doc)

    def unindex(self, doc):
        del self._doc_to_id[doc]
        Levenshtein_search.remove_string(self.index_key, doc)

    def initSearch(self):
        pass

    def search(self, doc, threshold=0):
        matching_docs = Levenshtein_search.lookup(self.index_key, doc, threshold)
        if matching_docs:
            return [self._doc_to_id[match] for match, _, _ in matching_docs]
        else:
            return []

    def __del__(self):
        Levenshtein_search.clear_wordset(self.index_key)
