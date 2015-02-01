#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import collections
import itertools
import logging
import time
import dedupe.tfidf as tfidf

logger = logging.getLogger(__name__)

    

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = None) :

        if stop_words is None :
            stop_words = defaultdict(set)

        self.predicates = predicates

        self.stop_words = stop_words

        self.tfidf_fields = defaultdict(set)

        for full_predicate in predicates :
            for predicate in full_predicate :
                if hasattr(predicate, 'canopy') :
                    self.tfidf_fields[predicate.field].add(predicate)

    #@profile
    def __call__(self, records):

        start_time = time.clock()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        for i, record in enumerate(records) :
            record_id, instance = record
    
            for pred_id, predicate in predicates :
                block_keys = predicate(record_id, instance)
                for block_key in block_keys :
                    yield block_key + pred_id, record_id
            
            if i and i % 10000 == 0 :
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                             {'iteration' :i,
                              'elapsed' :time.clock() - start_time})



    def _resetCanopies(self) :
        # clear canopies to reduce memory usage
        for predicate_set in self.tfidf_fields.values() :
            for predicate in predicate_set :
                predicate.canopy = {}


class DedupeBlocker(Blocker) :

    def tfIdfBlock(self, data, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        indices = {}
        for predicate in self.tfidf_fields[field] :
            index = tfidf.TfIdfIndex(field, self.stop_words[field])
            indices[predicate] = index

        base_tokens = {}

        for record_id, doc in data :
            base_tokens[record_id] = doc
            for index in indices.values() :
                index.index(record_id, doc)

        logger.info(time.asctime())                

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            index = indices[predicate]
            predicate.canopy = index.canopy(base_tokens, 
                                            predicate.threshold)
        
        logger.info(time.asctime())                
               
class RecordLinkBlocker(Blocker) :
    def tfIdfIndex(self, data_2, field): 
        '''Creates TF/IDF index of a given set of data'''
        predicate = next(iter(self.tfidf_fields[field]))

        index = predicate.index
        canopy = predicate.canopy

        if index is None :
            index = tfidf.TfIdfIndex(field, self.stop_words[field])
            canopy = {}

        for record_id, doc in data_2  :
            index.index(record_id, doc)
            canopy[record_id] = (record_id,)

        logger.info(time.asctime())                

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            predicate.index = index
            predicate.canopy = canopy

        logger.info(time.asctime())                

    def tfIdfUnindex(self, data_2, field) :
        '''Remove index of a given set of data'''
        predicate = next(iter(self.tfidf_fields[field]))

        index = predicate.index
        canopy = predicate.canopy

        for record_id, _ in data_2 :
            if record_id in canopy :
                index.unindex(record_id)
                del canopy[record_id]

        for predicate in self.tfidf_fields[field] :
            predicate.index = index
            predicate.canopy = canopy

