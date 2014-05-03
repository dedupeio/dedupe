#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import types
import logging
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.lexicon import Splitter
import time

import dedupe.tfidf as tfidf

logger = logging.getLogger(__name__)

    

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = defaultdict(set)) :
        
        self.predicates = predicates
        self.stop_words = stop_words
        self.canopies = defaultdict(dict)

        for compound_predicate in self.predicates :
            for predicate in compound_predicate :
                if predicate.type == "TfidfPredicate" :
                    self.canopies[predicate.field][predicate.threshold] = {}


    def __call__(self, records):

        for record in records :
            record_id = record[0]

            record_keys = set((':'.join(key), predicate)
                              for predicate 
                              in self.predicates
                              for key in predicate(record))

            for block_key in record_keys :
                yield block_key, record_id



class DedupeBlocker(Blocker) :
    def tfIdfBlock(self, data, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        splitter = Splitter()

        stop_word_remover = CustomStopWordRemover(self.stop_words[field])

        index = TextIndex(Lexicon(splitter, stop_word_remover))

        index.index = CosineIndex(index.lexicon)

        index_to_id = {}
        base_tokens = {}

        for i, (record_id, doc) in enumerate(data, 1) :
            index_to_id[i] = record_id
            base_tokens[i] = splitter.process([doc])
            index.index_doc(i, doc)

        canopies = (tfidf._createCanopies(index,
                                          base_tokens, 
                                          threshold, 
                                          field)
                    for threshold in self.tfidf_fields[field])

        for canopy in canopies :
            key, index_canopy = canopy
            id_canopy = dict((index_to_id[k], index_to_id[v]) 
                             for k,v in index_canopy.iteritems())
            self.canopies[key] = defaultdict(str, id_canopy)


class RecordLinkBlocker(Blocker) :
    def tfIdfBlock(self, data_1, data_2, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        splitter = Splitter()

        stop_word_remover = CustomStopWordRemover(self.stop_words[field])

        index = TextIndex(Lexicon(splitter, stop_word_remover))

        index.index = CosineIndex(index.lexicon)

        index_to_id = {}
        base_tokens = {}

        i = 1

        for record_id, doc in data_1 :
            index_to_id[i] = record_id
            base_tokens[i] = splitter.process([doc])
            i += 1

        for record_id, doc in data_2  :
            index_to_id[i] = record_id
            index.index_doc(i, doc)
            i += 1

        canopies = [apply(tfidf._createCanopies,
                          (index,
                           base_tokens, 
                           threshold, 
                           field))
                    for threshold in self.tfidf_fields[field]]


        for canopy in canopies :
            key, index_canopy = canopy
            id_canopy = dict((index_to_id[k], index_to_id[v]) 
                             for k,v in index_canopy.iteritems())
            self.canopies[key] = defaultdict(str, id_canopy)


def blockTraining(training_pairs,
                  predicate_set,
                  eta=.1,
                  epsilon=.1,
                  matching = "Dedupe"):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    # Setup

    training_dupes = (training_pairs['match'])[:]
    training_distinct = (training_pairs['distinct'])[:]

    if matching == "RecordLink" :
        coverage = RecordLinkCoverage(predicate_set,
                                      training_dupes + training_distinct)

    else :
        coverage = Coverage(predicate_set,
                            training_dupes + training_distinct)

    coverage_threshold = eta * len(training_distinct)
    logger.info("coverage threshold: %s", coverage_threshold)

    # Only consider predicates that cover at least one duplicate pair
    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               training_dupes)
    predicate_set = dupe_coverage.keys()

    # We want to throw away the predicates that puts together too
    # many distinct pairs
    distinct_blocks = coverage.predicateCoverage(predicate_set,
                                                 training_distinct)

    logger.info("Before removing liberal predicates, %s predicates",
                 len(predicate_set))

    for (pred, blocks) in distinct_blocks.iteritems():
        if any(len(block) >= coverage_threshold for block in blocks if block):
            predicate_set.remove(pred)

    logger.info("After removing liberal predicates, %s predicates",
                 len(predicate_set))

    distinct_coverage = coverage.predicateCoverage(predicate_set, 
                                                   training_distinct)

    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              distinct_coverage,
                                              epsilon,
                                              coverage)

    logger.info('Final predicate set:')
    for predicate in final_predicate_set :
        logger.info(predicate)

    if final_predicate_set:
        return final_predicate_set, coverage.stop_words
    else:
        raise ValueError('No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data')


def findOptimumBlocking(uncovered_dupes,
                        predicate_set,
                        distinct_coverage,
                        epsilon,
                        coverage):

    # Greedily find the predicates that, at each step, covers the
    # most duplicates and covers the least distinct pairs, due to
    # Chvatal, 1979
    #
    # We would like to optimize the ratio of the probability of of a
    # predicate covering a duplicate pair versus the probability of
    # covering a distinct pair. If we have a uniform prior belief
    # about those probabilities, we can estimate these probabilities as
    #
    # (predicate_covered_dupe_pairs + 1) / (all_dupe_pairs + 2)
    #
    # (predicate_covered_distinct_pairs + 1) / (all_distinct_pairs + 2)
    #
    # When we are trying to find the best predicate among a set of
    # predicates, the denominators factor out and our coverage
    # estimator becomes
    #
    # (predicate_covered_dupe_pairs + 1)/ (predicate_covered_distinct_pairs + 1)

    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               uncovered_dupes)
    
    uncovered_dupes = set(uncovered_dupes)

    final_predicate_set = []
    while len(uncovered_dupes) > epsilon:

        best_cover = 0
        best_predicate = None
        for predicate in dupe_coverage :
            dupes = len(dupe_coverage[predicate])
            distinct = len(distinct_coverage[predicate])
            cover = (dupes + 1.0)/(distinct + 1.0)
            if cover > best_cover:
                best_cover = cover
                best_predicate = predicate
                best_distinct = distinct
                best_dupes = dupes


        if not best_predicate:
            logger.warning('Ran out of predicates')
            break

        final_predicate_set.append(best_predicate)
        predicate_set.remove(best_predicate)
        
        uncovered_dupes = uncovered_dupes - dupe_coverage[best_predicate]
        dupe_coverage = coverage.predicateCoverage(predicate_set,
                                                   uncovered_dupes)


        logger.debug(best_predicate)
        logger.debug('cover: %(cover)f, found_dupes: %(found_dupes)d, '
                      'found_distinct: %(found_distinct)d, '
                      'uncovered dupes: %(uncovered)d',
                      {'cover' : best_cover,
                       'found_dupes' : best_dupes,
                       'found_distinct' : best_distinct,
                       'uncovered' : len(uncovered_dupes)
                       })

    return final_predicate_set

class Coverage(object) :
    def __init__(self, predicate_set, pairs) :
        self.overlapping = defaultdict(set)
        self.stop_words = {}


        covered_by = defaultdict(set)

        records = set(itertools.chain(*pairs))
        
        id_records = dict(itertools.izip(itertools.count(), records))
        record_ids = dict(itertools.izip(records, itertools.count()))

        blocker = DedupeBlocker(predicate_set) 

        for block_key, record_id in blocker(id_records.items()) :
            covered_by[record_id].add(block_key)

        for record_1, record_2 in pairs :
            id_1 = record_ids[record_1]
            id_2 = record_ids[record_2]
            
            blocks = covered_by[id_1] & covered_by[id_2] 
            for block_key, predicate in blocks :
                self.overlapping[predicate].add((record_1, record_2))

    def predicateCoverage(self,
                          predicate_set,
                          pairs) :

        coverage = defaultdict(set)
        pairs = set(pairs)

        for predicate in predicate_set :
            covered_pairs = pairs.intersection(self.overlapping[predicate])
            if covered_pairs :
                coverage[predicate] = covered_pairs

        return coverage



def stopWords(data) :
    index = TextIndex(Lexicon(Splitter()))

    for i, (_, doc) in enumerate(data, 1) :
        index.index_doc(i, doc)

    doc_freq = [(len(index.index._wordinfo[wid]), word) 
                for word, wid in index.lexicon.items()]

    doc_freq.sort(reverse=True)

    N = float(index.index.documentCount())
    threshold = int(max(1000, N * 0.05))

    stop_words = set([])

    for frequency, word in doc_freq :
        if frequency > threshold :
            stop_words.add(word)
        else :
            break

    return stop_words

class Predicate(object) :
    def __repr__(self) :
        return "%s: %s" % (self.type, self.__name__)


class SimplePredicate(Predicate) :
    type = "SimplePredicate"

    def __init__(self, func, field) :
        self.func = func
        self.__name__ = "(%s, %s)" % (func.__name__, field)
        self.field = field

    def __call__(self, instance) :
        record = instance[1]
        for block_key in  self.func(record[self.field]) :
            yield block_key

class TfidfPredicate(Predicate):
    type = "TfidfPredicate"

    def __init__(self, threshold, field):
        self.__name__ = '(%s, %s)' % (threshold, field)
        self.field = field
        self.canopy = defaultdict(int)
        self.threshold = threshold

    def __call__(self, record) :
        record_id = record[0]
        center = self.canopy[record_id]
        if center :
            return (unicode(center),)
        else :
            return ()



class CompoundPredicate(Predicate) :
    type = "CompoundPredicate"

    def __init__(self, predicates) :
        self.predicates = predicates
        self.__name__ = '(%s)' % ', '.join([str(pred)
                                            for pred in 
                                            predicates])

    def __iter__(self) :
        for pred in self.predicates :
            yield pred

    def __call__(self, record) :
        block_keys = []
        for predicate in self.predicates :
            block_keys.append(list(predicate(record)))
            
        for block_key in itertools.product(*block_keys) :
            yield block_key

class CustomStopWordRemover(object):
    def __init__(self, stop_words) :
        self.stop_words = stop_words.copy()

    def process(self, lst):
        return [w for w in lst if not w in self.stop_words]

