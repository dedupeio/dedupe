#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import collections
import itertools
import logging
import time

logger = logging.getLogger(__name__)

    

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = None) :

        if stop_words is None :
            stop_words = defaultdict(lambda : defaultdict(set))

        self.predicates = predicates

        self.stop_words = stop_words

        self.index_fields = defaultdict(lambda:defaultdict(set))

        for full_predicate in predicates :
            for predicate in full_predicate :
                if hasattr(predicate, 'index') :
                    self.index_fields[predicate.field][predicate.type].add(predicate)

    #@profile
    def __call__(self, records):

        start_time = time.clock()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        

        for i, record in enumerate(records) :
            record_id, instance = record

            for pred_id, predicate in predicates :
                block_keys = predicate(instance)
                for block_key in block_keys :
                    yield block_key + pred_id, record_id
            
            if i and i % 10000 == 0 :
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                             {'iteration' :i,
                              'elapsed' :time.clock() - start_time})



    def resetIndices(self) :
        # clear canopies to reduce memory usage
        for index_type in self.index_fields.values() :
            for predicate in list(index_type.values())[0] :
                predicate.index = None
                if hasattr(predicate, 'canopy') :
                    predicate.canopy = {}

    def index(self, data, field): 
        '''Creates TF/IDF index of a given set of data'''
        indices = extractIndices(self.index_fields[field],
                                 self.stop_words[field])

        for doc in data :
            for _, index, preprocess in indices :
                index.index(preprocess(doc))

        for index_type, index, _ in indices :

            index.initSearch()

            for predicate in self.index_fields[field][index_type] :
                logger.info("Canopy: %s", str(predicate))
                predicate.index = index

    def unindex(self, data, field) :
        '''Remove index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data :
            for _, index, preprocess in indices :
                index.unindex(preprocess(doc))

        for index_type, index, _ in indices :

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type] :
                logger.info("Canopy: %s", str(predicate))
                predicate.index = index


def extractIndices(index_fields, stop_words=None) :
    
    indices = []
    for index_type, predicates in index_fields.items() :
        predicate = next(iter(predicates))
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None :
            index = predicate.initIndex(stop_words[index_type])
        indices.append((index_type, index, preprocess))

    return indices

