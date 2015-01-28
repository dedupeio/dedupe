from zope.index.text.lexicon import Lexicon
from zope.index.text.stopdict import get_stopdict
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from BTrees.Length import Length
import re

class CanopyIndex(TextIndex) : # pragma : no cover
    def __init__(self, stop_words) : 
        lexicon = CanopyLexicon(stop_words)
        self.index = CosineIndex(lexicon)
        self.lexicon = lexicon


class CanopyLexicon(Lexicon) : # pragma : no cover
    def __init__(self, stop_words) : 
        super(CanopyLexicon, self).__init__()
        self._pipeline = [Splitter(),
                          CustomStopWordRemover(stop_words),
                          OperatorEscaper()]

    def sourceToWordIds(self, doc): 
        if doc is None:
            doc = ''
        last = stringify(doc) # this is changed line
        for element in self._pipeline:
            last = element.process(last)
        if not isinstance(self.wordCount, Length):
            self.wordCount = Length(self.wordCount())
        self.wordCount._p_deactivate()
        return list(map(self._getWordIdCreate, last))

    def isGlob(self, word) :
        return False


class CustomStopWordRemover(object):
    def __init__(self, stop_words) :
        self.stop_words = set(get_stopdict().keys())
        self.stop_words.update(stop_words)

    def process(self, lst):
        return [w for w in lst if not w in self.stop_words]


class OperatorEscaper(object) :
    def __init__(self) :
        self.operators = {"AND"  : "\AND",
                          "OR"   : "\OR",
                          "NOT"  : "\NOT",
                          "("    : "\(",
                          ")"    : "\)",
                          "ATOM" : "\ATOM",
                          "EOF"  : "\EOF"}

    def process(self, lst):
        return [self.operators.get(w.upper(), w) for w in lst]


def stringify(doc) :
    if not isinstance(doc, basestring) :
        doc = u' '.join(u'_'.join(each.split()) for each in doc)

    return [doc]



class Splitter(object):
    rx = re.compile(r"(?u)\w+[\w*?]*")

    def process(self, lst):
        result = []
        for s in lst:
            result += self.rx.findall(s)

        return result
