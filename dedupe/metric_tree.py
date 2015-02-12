from finenight import iadfa
from finenight import possibleStates
from finenight import fsc 

DISTANCE=2

class LevenshteinIndex(object) :
    def __init__(self, stop_words=[]) :

        self._index = iadfa.IncrementalAdfa()

    def initSearch(self) :
        self._index.initSearch()

        transition_states = possibleStates.genTransitions(DISTANCE)

        self.etr = fsc.ErrorTolerantRecognizer(DISTANCE, 
                                               self._index, 
                                               transition_states)
        

    def index(self, record) :
        self._index.createFromSortedListOfWords(record)

    def search(self, doc, threshold=0) :
        return self.etr.recognize(doc)
