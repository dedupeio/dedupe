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


def stopWords(inverted_index, stop_word_threshold) :
    stop_words= set([])

    for atom in inverted_index.atoms() :
        df = inverted_index.getDF(atom)
        if df < 2 :
            stop_words.add(atom)
        elif df > stop_word_threshold :
            stop_words.add(atom)
    
    return stop_words

        
def weightVectors(weight_vectors, tokenized_records, stop_words) :
    weighted_records = {}

    for record_id, vector in tokenized_records.iteritems() :
        weighted_vector = weight_vectors[vector]
        weighted_vector.name = vector.name
        for atom in weighted_vector :
            if atom in stop_words :
                del weighted_vector[atom]
        if weighted_vector :
            weighted_records[record_id] = weighted_vector

    return weighted_records

def tokensToInvertedIndex(token_vectors) :
    i_index = defaultdict(set)
    for record_id, vector in token_vectors.iteritems() :
        for token in vector :
            i_index[token].add(vector)
    
    return i_index
            


def fieldToAtomVector(field, record_id, tokenfactory) :
    tokens = words.findall(field.lower())
    av = mk.AtomVector(name=record_id)
    for token in tokens :
        av[tokenfactory[token]] += 1
    
    return av

def unweightedIndex(data, fields) :    

    tokenfactory = mk.AtomFactory("tokens")  
    tokenized_records = defaultdict(dict)
    tokenized_center_records = defaultdict(dict)
    inverted_indices = defaultdict(lambda : mk.InvertedIndex())
  
    for record_id, record in data:
        for field in fields:
            av = fieldToAtomVector(record[field], record_id, tokenfactory)
            inverted_indices[field].add(av) 
            if record.constrained :
                tokenized_center_records[field][record_id] = av
            else : 
                tokenized_records[field][record_id] = av
                
                

    return tokenized_records, tokenized_center_records, inverted_indices

def invertIndex(data, fields):
    (tokenized_records, 
     tokenized_center_records,
     inverted_indices) = unweightedIndex(data, fields)

    num_docs = inverted_indices.values()[0].getN()
    stop_word_threshold = max(num_docs * 0.025, 500)
    logging.info('Stop word threshold: %(stop_thresh)d',
                 {'stop_thresh' :stop_word_threshold})

    weighted_records_d = defaultdict(dict)
    weighted_center_records_d = defaultdict(dict)
    weighted_inverted_indices = {}

    for field in fields :
        inverted_index = inverted_indices[field]

        weighted_vectors = mk.WeightVectors(inverted_index)
        stop_words = stopWords(inverted_index, stop_word_threshold)

        weighted_records = weightVectors(weighted_vectors,
                                         tokenized_records[field],
                                         stop_words)

        weighted_inverted_indices[field] = tokensToInvertedIndex(weighted_records)
        weighted_records_d[field] = weighted_records

        if tokenized_center_records :
            weighted_center_records_d[field] = weightVectors(weighted_vectors,
                                                             tokenized_center_records[field],
                                                             stop_words)


    if tokenized_center_records :
        return weighted_inverted_indices, weighted_center_records_d

    else :
        return weighted_inverted_indices, weighted_records_d


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
                corpus_ids.difference_update([candidate_vector.name])

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
