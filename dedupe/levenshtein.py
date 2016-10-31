import collections
import itertools
import Levenshtein_search
from .index import Index

class LevenshteinIndex(Index):
    def __init__(self):
        self.index_key = Levenshtein_search.populate_wordset(-1, [])

        try : # py 2
            self._doc_to_id = collections.defaultdict(itertools.count(1).next)
        except AttributeError : # py 3
            self._doc_to_id = collections.defaultdict(itertools.count(1).__next__)

        self.docs = []

    def index(self, doc):
        i = self._doc_to_id[doc]
        self.docs.append(doc)

    def unindex(self, doc):
        pass

    def initSearch(self) :
        Levenshtein_search.populate_wordset(self.index_key,
                                            self.docs)
        self.docs = []

    def search(self, doc, threshold=0):
        results = Levenshtein_search.lookup(self.index_key, doc, threshold)
        
        return [doc for doc, _, _ in results]
        

        

        
        
