#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from zope.index.text.parsetree import ParseError

logger = logging.getLogger(__name__)

#@profile
def makeCanopy(index, token_vector, threshold) :
    canopies = {}
    seen = set([])
    corpus_ids = set(token_vector.keys())

    while corpus_ids:
        center_id = corpus_ids.pop()
        center_vector = token_vector[center_id]

        index.unindex_doc(center_id)
        
        if not center_vector :
            continue

        try :
            search_string = ' OR '.join(center_vector)
            candidates = index.apply(search_string).byValue(threshold)
        except ParseError :
            continue

        candidates = set(k for  _, k in candidates)

        corpus_ids.difference_update(candidates)

        for candidate_id in candidates :
            canopies[candidate_id] = center_id
            index.unindex_doc(candidate_id)

        if candidates :
            canopies[center_id] = center_id

    return canopies

