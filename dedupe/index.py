from __future__ import division

from zope.index.text.lexicon import Lexicon
from zope.index.text.stopdict import get_stopdict
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.setops import mass_weightedUnion

from BTrees.Length import Length
import re
import math
import numpy

class CanopyIndex(TextIndex) : # pragma : no cover
    def __init__(self, stop_words) : 
        lexicon = CanopyLexicon(stop_words)
        self.index = CosineIndex(lexicon)
        self.lexicon = lexicon

    def initSearch(self) :
        N = len(self.index._docweight)

        self._wids_dict = {}

        bucket = self.index.family.IF.Bucket
        for wid, docs in self.index._wordinfo.items() :
            if isinstance(docs, dict) :
                docs = bucket(docs)
            idf = numpy.log1p(N/len(docs))
            self.index._wordinfo[wid] = docs
            term = self.lexicon._words[wid]
            self._wids_dict[term] = (wid, idf)


    def apply(self, query_list, threshold, start=0, count=None):
        _wids_dict = self._wids_dict
        _wordinfo = self.index._wordinfo
        l_pow = float.__pow__

        L = []
        qw = 0

        for term in query_list :
            wid, weight = _wids_dict.get(term, (None, None))
            if wid is None :
                continue
            docs = _wordinfo[wid]
            L.append((docs, weight))
            qw += l_pow(weight, 2)

        results = mass_weightedUnion(L)

        qw = math.sqrt(qw)
        results = results.byValue(qw * threshold)

        return results


class CanopyLexicon(Lexicon) : # pragma : no cover
    def __init__(self, stop_words) : 
        super(CanopyLexicon, self).__init__()
        self._pipeline = [CustomStopWordRemover(stop_words)]

    def sourceToWordIds(self, last): 
        if last is None:
            last = [] 
        for element in self._pipeline:
            last = element.process(last)
        if not isinstance(self.wordCount, Length):
            self.wordCount = Length(self.wordCount())
        self.wordCount._p_deactivate()
        return list(map(self._getWordIdCreate, last))

class CustomStopWordRemover(object):
    def __init__(self, stop_words) :
        self.stop_words = set(get_stopdict().keys())
        self.stop_words.update(stop_words)

    def process(self, lst):
        return [w for w in lst if not w in self.stop_words]


