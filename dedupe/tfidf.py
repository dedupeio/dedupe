#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import logging
import re
import dedupe.mekano as mk
from zope.index.text.parsetree import ParseError

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

def removeStopWords(inverted_index, filter_frequent_words) :
    wv = mk.WeightVectors(inverted_index)

    atoms = ((inverted_index.getDF(atom), atom) 
             for atom in inverted_index.atoms())

    frequent_tokens = []

    if filter_frequent_words :

        N_records = float(inverted_index.getN())

        threshold = min(int(0.1 * N_records), 1000)
        logging.info('Stop word threshold: %(stop_thresh)d',
                     {'stop_thresh' : threshold})


        for n_records, atom in atoms :
            if n_records > threshold :
                frequent_tokens.append(atom)
                inverted_index.clear(atom)

            if n_records < 2 :
                inverted_index.clear(atom)

    else :

        for n_records, atom in atoms :
            if n_records < 2 :
                inverted_index.clear(atom)

    for atom in inverted_index.atoms() :
        for av in inverted_index.getii(atom) :
            av.Normalize()

    return frequent_tokens


    

        



def tokensToInvertedIndex(token_vectors) :
    i_index = defaultdict(set)

    n_words = 0

    for record_id, vector in token_vectors.iteritems() :
        for token in vector :
            i_index[token].add(vector)
            n_words += 1

    for token, vectors in i_index.items() :
        if len(vectors) < 2 :
            del i_index[token]

    percentage_tokens = 0

    for k in sorted(i_index.items(), key=lambda i : -len(i[1]) ) :
        del i_index[k[0]]
        percentage_tokens += len(k[1])/float(n_words)
        if percentage_tokens > 0.4 :
            break
        
    print "hello"
    print len(i_index)
    
    return i_index
            




class InvertedIndex(object) :
    def __init__(self, stop_words) :
        self.tokenfactory = mk.AtomFactory("tokens")  
        self.stop_words = stop_words
        self.inverted_index = mk.InvertedIndex()
            
    def unweightedVectors(self, data) :
        tokenized_records = {}

        for record_id, record in data:
            av = self.fieldToAtomVector(record, record_id) 
            self.inverted_index.add(av) 
            tokenized_records[record_id] = av
                
        return tokenized_records
        
    def stopWordThreshold(self) :
        num_docs = self.inverted_indices.values()[0].getN()
        self.stop_word_threshold = max(num_docs * 0.025, 500)
        logging.info('Stop word threshold: %(stop_thresh)d',
                     {'stop_thresh' :self.stop_word_threshold})


    def fieldToAtomVector(self, field, record_id) :
        tokens = words.findall(field.lower())
        av = mk.AtomVector(name=record_id)
        for token in tokens :
            if token not in self.stop_words :
                av[self.tokenfactory[token]] += 1

        return av






#@profile
def makeCanopy(index, token_vector, threshold) :
    canopies = defaultdict(str)
    seen = set([])
    corpus_ids = set(token_vector.keys())

    while corpus_ids:
        center_id = corpus_ids.pop()
        canopies[center_id] = center_id
        center_vector = token_vector[center_id]
        
        seen.add(center_vector)


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

    return canopies

def _createCanopies(field_inverted_index,
                    token_vector,
                    threshold,
                    field) : 

    canopy = makeCanopy(field_inverted_index, token_vector, threshold)

    logging.info("Canopy: %s", threshold.__name__ + field)
    return {(threshold, field) :  canopy}

    

    
    
