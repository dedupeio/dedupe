#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import collections
import itertools
import logging
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.lexicon import Splitter
from zope.index.text.stopdict import get_stopdict
import time
import dedupe.tfidf as tfidf

logger = logging.getLogger(__name__)

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = None) :

        if stop_words is None :
            stop_words = defaultdict(lambda : set(get_stopdict()))

        self.predicates = predicates

        self.stop_words = stop_words

        self.tfidf_fields = defaultdict(set)

        for full_predicate in predicates :
            for predicate in full_predicate :
                if hasattr(predicate, 'canopy') :
                    self.tfidf_fields[predicate.field].add(predicate)

    #@profile
    def __call__(self, records):

        start_time = time.time()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        for i, record in enumerate(records) :
            record_id, instance = record
    
            for pred_id, predicate in predicates :
                block_keys = predicate(record_id, instance)
                for block_key, rec_id in block_keys :
                    yield block_key + pred_id, rec_id
            
            if i and i % 10000 == 0 :
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                             {'iteration' :i,
                              'elapsed' :time.time() - start_time})



    def _resetCanopies(self) :
        # clear canopies to reduce memory usage
        for predicate_set in self.tfidf_fields.values() :
            for predicate in predicate_set :
                predicate.canopy = {}
                predicate.index = None
                predicate.index_to_id = None

    def _lexicon(self, field) :
        splitter = Splitter()
        stop_word_remover = CustomStopWordRemover(self.stop_words[field])
        operator_escaper = OperatorEscaper()
        
        return Lexicon(splitter, stop_word_remover, operator_escaper)

class DedupeBlocker(Blocker) :

    def tfIdfBlock(self, data, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        indices = {}
        for predicate in self.tfidf_fields[field] :
            index = TextIndex(self._lexicon(field))
            index.index = CosineIndex(index.lexicon)
            indices[predicate] = index

        parseTerms = index.lexicon.parseTerms
        stringify = predicate.stringify

        index_to_id = {}
        base_tokens = {}

        for i, (record_id, doc) in enumerate(data, 1) :
            doc = stringify(doc)
            index_to_id[i] = record_id
            base_tokens[i] = ' OR '.join(parseTerms(doc))
            for index in indices.values() :
                index.index_doc(i, doc)

        logger.info(time.asctime())                

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            canopy = tfidf.makeCanopy(indices[predicate],
                                      base_tokens, 
                                      predicate.threshold)
            predicate.canopy = dict((index_to_id[k], index_to_id[v])
                                    for k, v
                                    in canopy.iteritems())
        
        logger.info(time.asctime())                
               
class RecordLinkBlocker(Blocker) :
    def tfIdfIndex(self, data_2, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        indices = {}
        for predicate in self.tfidf_fields[field] :
            index = TextIndex(self._lexicon(field))
            index.index = CosineIndex(index.lexicon)
            indices[predicate] = index

        stringify = predicate.stringify

        index_to_id = {}

        for i, (record_id, doc) in enumerate(data_2)  :
            doc = stringify(doc)
            index_to_id[i] = record_id
            for index in indices.values() :
                index.index_doc(i, doc)

        for predicate in self.tfidf_fields[field] :
            predicate.index = indices[predicate]
            predicate.index_to_id = index_to_id
            predicate.canopy = dict((v, v) for v in index_to_id.values())


class CustomStopWordRemover(object):
    def __init__(self, stop_words) :
        self.stop_words = stop_words

    def process(self, lst):
        return [w for w in lst if not w in self.stop_words]


class OperatorEscaper(object) :
    def __init__(self) :
        self. operators = {"AND"  : "\AND",
                           "OR"   : "\OR",
                           "NOT"  : "\NOT",
                           "("    : "\(",
                           ")"    : "\)",
                           "ATOM" : "\ATOM",
                           "EOF"  : "\EOF"}

    def process(self, lst):
        return [self.operators.get(w, w) for w in lst]
