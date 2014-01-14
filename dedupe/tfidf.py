#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from zope.index.text.parsetree import ParseError

class TfidfPredicate(float):
    def __new__(self, threshold):
        return float.__new__(self, threshold)

    def __init__(self, threshold):
        self.__name__ = 'TF-IDF:' + str(threshold)

    def __repr__(self) :
        return self.__name__

#@profile
def makeCanopy(index, token_vector, threshold, dedupe=True) :
    canopies = {}
    seen = set([])
    corpus_ids = set(token_vector.keys())

    while corpus_ids:
        center_id = corpus_ids.pop()
        center_vector = token_vector[center_id]

        seen.add(center_id)
        
        if not center_vector :
            continue

        try :
            candidates = index.apply('"%s"' % center_vector).byValue(threshold)
        except ParseError :
            continue

        candidates = set(k for  _, k in candidates) - seen

        seen.update(candidates)
        corpus_ids.difference_update(candidates)

        for candidate_id in candidates :
            canopies[candidate_id] = center_id

        if candidates and dedupe :
            canopies[center_id] = center_id


    return canopies

def _createCanopies(field_inverted_index,
                    token_vector,
                    threshold,
                    field,
                    dedupe=True) : 

    canopy = makeCanopy(field_inverted_index, token_vector, threshold, dedupe)

    logging.info("Canopy: %s", threshold.__name__ + field)
    return ((threshold, field),  canopy)

    

    
    
