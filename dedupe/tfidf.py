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
 
        self._i_to_id = {}
        
        self._id_to_i = collections.defaultdict(itertools.count(-2**31).next)
        
        self._parseTerms = self._index.lexicon.parseTerms

    def index(self, record_id, doc) :
        i = self._id_to_i[record_id]
        self._i_to_id[i] = record_id

        try :
            self._index.index_doc(i, doc)
        except :
            print doc
            raise

    def unindex(self, record_id) :
        i = self._id_to_i.pop(record_id)
        del self._i_to_id[i]
        self._index.unindex_doc(i)

    def search(self, doc, threshold=0) :
        query_list = self._parseTerms(doc)
        query = ' OR '.join(query_list)

        if query :
            results = self._index.apply(query).byValue(threshold)
        else :
            results = []

        return [self._i_to_id[k] 
                for  _, k in results]

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


