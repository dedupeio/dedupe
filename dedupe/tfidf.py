#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from index import CanopyIndex
import math
import collections
import itertools

logger = logging.getLogger(__name__)

class TfIdfIndex(object) :
    def __init__(self, field, stop_words=[]) :
        self.field = field

        self._index = CanopyIndex(stop_words)
 
        self._id_to_i = collections.defaultdict(itertools.count(1).next)
        
        self._parseTerms = self._index.lexicon.parseTerms

    def index(self, record_id, doc) :
        i = self._id_to_i[record_id]
        self._index.index_doc(i, doc)

    def unindex(self, record_id) :
        i = self._id_to_i.pop(record_id)
        self._index.unindex_doc(i)

    def search(self, doc, threshold=0) :
        query_list = self._parseTerms(doc)
 
        if query_list :
            results = [center for score, center 
                       in self._index.apply(query_list, threshold)]
        else :
            results = []

        return results

