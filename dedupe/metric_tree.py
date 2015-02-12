from finenight import iadfa
from finenight import fsc 
from finenight.fsc import transitions

class LevenshteinIndex(object) :
    def __init__(self, stop_words=[]) :

        self._index = iadfa.IncrementalAdfa()

    def initSearch(self) :
        self._index.initSearch()
        self.etr = fsc.ErrorTolerantRecognizer(self._index)

    def index(self, record) :
        self._index.createFromSortedListOfWords(record)

    def search(self, doc, transitions, n=1) :
        return self.etr.recognize(doc, transitions, n)


