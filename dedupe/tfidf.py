#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import logging
import re
import mekano as mk

words = re.compile("[\w']+")

class TfidfPredicate(float):
    def __new__(self, threshold):
        return float.__new__(self, threshold)

    def __init__(self, threshold):
        self.__name__ = 'TF-IDF:' + str(threshold)

def invertIndex(data, tfidf_fields, df_index=None):

    tokenfactory = mk.AtomFactory("tokens")
    
    inverted_index = {}
    for field in tfidf_fields :
        inverted_index[field] = mk.InvertedIndex()

    token_vector = defaultdict(dict)

    for (record_id, record) in data:
        for field in tfidf_fields:
            tokens = words.findall(record[field].lower())
            av = mk.AtomVector(name=record_id)
            for token in tokens :
                av[tokenfactory[token]] += 1
            inverted_index[field].add(av)

            token_vector[field][record_id] = av


    return (inverted_index, token_vector)

def createCanopies(field,
                   threshold,
                   token_vector,
                   inverted_index):
    """
    A function that returns a field value of a record with a
    particular doc_id, doc_id is the only argument that must be
    accepted by select_function
    """

    canopies = defaultdict(lambda : None)
    seen_set = set([])

    field_inverted_index = inverted_index[field]

    token_vectors = token_vector[field]
    corpus_ids = token_vectors.keys()

    weight_vector = mk.WeightVectors(field_inverted_index, cache=True)

    while corpus_ids:
        center_id = corpus_ids.pop()
        canopies[center_id] = center_id

        center_vector = token_vectors[center_id]
        center_norm = center_vector.CosineLen()

        seen_set.add(center_vector)

        candidate_set = set()

        for token in center_vector :
            for doc in field_inverted_index.getii(token) :
                candidate_set.add(doc)

        candidate_set = candidate_set - seen_set

        w_center_vector = weight_vector[center_vector]

        for candidate_vector in candidate_set:
            w_candidate_vector = weight_vector[candidate_vector]

            similarity = (w_candidate_vector * w_center_vector)         

            if similarity > threshold :
                canopies[candidate_vector.name] = center_id
                seen_set.add(candidate_vector)
                corpus_ids.remove(candidate_vector.name)

    return canopies
