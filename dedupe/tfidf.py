#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import logging
import re
import dedupe.mekano as mk

words = re.compile("[\w']+")

class TfidfPredicate(float):
    def __new__(self, threshold):
        return float.__new__(self, threshold)

    def __init__(self, threshold):
        self.__name__ = 'TF-IDF:' + str(threshold)

    def __repr__(self) :
        return self.__name__


def weightVectors(inverted_index, token_vectors, stop_word_threshold) :


    for field in token_vectors :
        singletons = set([])
        stop_words = set([])
        for atom in inverted_index[field].atoms() :
            df = inverted_index[field].getDF(atom)
            if df < 2 :
                singletons.add(atom)
            elif df > stop_word_threshold :
                stop_words.add(atom)
                
        

        wv = mk.WeightVectors(inverted_index[field])
        ii = defaultdict(set)
        for record_id, vector in token_vectors[field].iteritems() :
            w_vector = wv[vector]
            w_vector.name = vector.name
            for atom in w_vector :
                if atom in singletons or atom in stop_words :
                    del w_vector[atom]
            token_vectors[field][record_id] = w_vector
            for token in w_vector :
                ii[token].add(w_vector)
            
            

        inverted_index[field] = ii

    return token_vectors, inverted_index

def invertIndex(data, tfidf_fields, df_index=None):

    tokenfactory = mk.AtomFactory("tokens")  
    inverted_index = {}

    for field in tfidf_fields :
        inverted_index[field] = mk.InvertedIndex()

    token_vector = defaultdict(dict)

    for record_id, record in data:
        for field in tfidf_fields:
            tokens = words.findall(record[field].lower())
            av = mk.AtomVector(name=record_id)
            for token in tokens :
                av[tokenfactory[token]] += 1
            inverted_index[field].add(av)

            token_vector[field][record_id] = av

    num_docs = inverted_index.values()[0].getN()

    stop_word_threshold = max(num_docs * 0.025, 500)
    logging.info('Stop word threshold: %(stop_thresh)d',
                 {'stop_thresh' :stop_word_threshold})


    token_vectors, inverted_index = weightVectors(inverted_index, 
                                                  token_vector,
                                                  stop_word_threshold)
    

    return (inverted_index, token_vector)

#@profile
def makeCanopy(inverted_index, token_vector, threshold) :
    canopies = defaultdict(lambda:None)
    seen = set([])
    corpus_ids = set(token_vector.keys())


    while corpus_ids:
        center_id = corpus_ids.pop()
        canopies[center_id] = center_id
        center_vector = token_vector[center_id]
        
        seen.add(center_vector)

        if not center_vector :
            continue
     
        candidates = set.union(*(inverted_index[token] 
                                 for token in center_vector))

        candidates = candidates - seen

        for candidate_vector in candidates :

            similarity = candidate_vector * center_vector         

            if similarity > threshold :
                candidate_id = candidate_vector.name
                canopies[candidate_id] = center_id
                seen.add(candidate_vector)
                corpus_ids.remove(candidate_id)

    return canopies

    

def createCanopies(field,
                   threshold,
                   token_vector,
                   inverted_index):
    """
    A function that returns a field value of a record with a
    particular doc_id, doc_id is the only argument that must be
    accepted by select_function
    """

    field_inverted_index = inverted_index[field]
    token_vectors = token_vector[field]

    return makeCanopy(field_inverted_index, token_vectors, threshold)
