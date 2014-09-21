#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.lexicon import Splitter
from zope.index.text.stopdict import get_stopdict
from zope.index.text.parsetree import ParseError
import math

logger = logging.getLogger(__name__)

class TfIdfIndex(object) :
    def __init__(self, field, stop_words=[]) :
        self.field = field
 
        splitter = Splitter()
        stop_word_remover = CustomStopWordRemover(stop_words)
        operator_escaper = OperatorEscaper()
        lexicon = Lexicon(splitter, stop_word_remover, operator_escaper)

        self._index = TextIndex(lexicon)
        self._index.index = CosineIndex(self._index.lexicon)

        self._i_to_id = {}
        self._parseTerms = self._index.lexicon.parseTerms

    def _hash(self, x) :
        i = hash(x)
        return int(math.copysign(i % (2**31), i))
        

    def index(self, record_id, doc) :
        i = self._hash(record_id)
        self._i_to_id[i] = record_id

        self._index.index_doc(i, doc)

    def unindex(self, record_id) :
        i = self._hash(record_id)
        del self._i_to_id[i]
        self._index.unindex_doc(i)

    def search(self, doc, threshold=0) :
        doc = self._stringify(doc)
        query_list = self._parseTerms(doc)
        query = ' OR '.join(query_list)

        if query :
            results = self._index.apply(query).byValue(threshold)
        else :
            results = []

        return [self._i_to_id[k] 
                for  _, k in results]

    def _stringify(self, doc) :
        try :
            doc = u' '.join(u'_'.join(each.split() for each in doc))
        except TypeError :
            pass

        return doc

    def canopy(self, token_vector, threshold) :
        canopies = {}
        seen = set([])
        corpus_ids = set(token_vector.keys())

        while corpus_ids:
            center_id = corpus_ids.pop()
            center_vector = token_vector[center_id]

            self.unindex(center_id)
        
            if not center_vector :
                continue

            candidates = self.search(center_vector, threshold)
            
            candidates = set(candidates)

            corpus_ids.difference_update(candidates)

            for candidate_id in candidates :
                canopies[candidate_id] = (center_id,)
                self.unindex(candidate_id)

            if candidates :
                canopies[center_id] = (center_id,)

        return canopies


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


