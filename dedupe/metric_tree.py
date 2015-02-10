from finenight import iadfa
from finenight import recognize
from finenight import fsc 

DISTANCE=3

class LevenshteinIndex(object) :
    def __init__(self, stop_words=[]) :
        transition_states = recognize.getTransitionStates("finenight/levenshtein.dat",
                                                               DISTANCE)

        self.etr = fsc.ErrorTolerantRecognizer(DISTANCE, transition_states)
        self._index = iadfa.IncrementalAdfa()

    def index(self, record) :
        print record
        self._index.createFromSortedListOfWords(record)

    def search(self, doc, threshold=0) :
        return self.etr.recognize(doc, self._index)
