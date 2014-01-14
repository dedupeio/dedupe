#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import types
import logging
from multiprocessing import Pool
import mekano
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.lexicon import Splitter
import time

import tfidf
from backport import OrderedDict

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, predicates = None, 
                 pool=None,
                 stop_words = {}) :

        self.canopies = defaultdict(dict)

        self.pool = pool
        self.stop_words = defaultdict(set, stop_words)

        if predicates is None :
            self.simple_predicates = set([])
            self.tfidf_predicates = set([])
            self.tfidf_fields = []

        else :
            predicate_types = predicateTypes(predicates)
            self.simple_predicates, self.tfidf_predicates = predicate_types

            self.tfidf_fields = {}
            for threshold, field in self.tfidf_predicates :
                try :
                    self.tfidf_fields[field].add(threshold)
                except KeyError :
                    self.tfidf_fields[field] = set([threshold])

        self.predicates = predicates


    def functional(self, predicate) :
        F, field = predicate

        if F.__class__ is tfidf.TfidfPredicate :
            canopy = self.canopies[(F, field)]
            def tfidf_functional(instance) :
                record_id = instance[0]
                center = canopy[record_id]
                if center :
                    return (unicode(center),)
                else :
                    return ()
                        
            return tfidf_functional


        else :
            def simple_functional(instance) :
                record = instance[1]
                return F(record[field])

            return simple_functional 
                                                    

    def __call__(self, records):
        if self.tfidf_predicates and not self.canopies :
            raise ValueError("No canopies defined, but tf-idf predicate "
                             "learned. Did you run the tfIdfBlocks method "
                             "of the blocker?")

        _join = str.join
        _product = itertools.product

        functional_predicates = []
        for i, predicate in enumerate(self.predicates) :
            functionals = []
            for pred in predicate :
                functionals.append(self.functional(pred))

            functional_predicates.append((':' + unicode(i),
                                          functionals))

        start_time = time.time()

        for i, record in enumerate(records) :
            record_id = record[0]
            record_keys = set(_join(':', key) + label 
                              for label, predicate 
                              in functional_predicates
                              for key in _product(*[F(record) 
                                                    for F in predicate]))
            for key in record_keys :
                yield (key, record_id)

            if i % 10000 == 0 :
                print i, ',', time.time() - start_time, 'seconds'





class DedupeBlocker(Blocker) :
    def tfIdfBlock(self, data, field, filter_frequent_tokens=False): 
        '''Creates TF/IDF canopy of a given set of data'''
        
        index = TextIndex(Lexicon(Splitter()))

        index.index = CosineIndex(index.lexicon)

        def indexing_gen() :
            for record_id, doc in data :
                record_id = int(record_id)
                index.index_doc(record_id, doc)
                yield (record_id, doc)
        
        base_tokens = dict(indexing_gen())

        canopies = (apply(tfidf._createCanopies,
                          (index,
                           base_tokens, 
                           threshold, 
                           field))
                    for threshold in self.tfidf_fields[field])

        for canopy in canopies :
            self.canopies.update(canopy)

        # TextIndex is a highly recursive data structure, and there
        # are some hard limits in our ability to pickle such an object.
        # Since multiprocessing depends upon pickling, when we have
        # bigger data, we can't parallelize the construction of 
        # canopies. 
        #
        # http://stackoverflow.com/q/2134706/98080
        #
        # recursion_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(100000)
        #
        # results = [self.pool.apply_async(tfidf._createCanopies,
        #                                  (index,
        #                                   base_tokens, 
        #                                   threshold, 
        #                                   field),
        #                                  callback=self.canopies.update)
        #            for threshold in self.tfidf_fields[field]]
        #
        # for r in results :
        #     r.wait()
        #
        # sys.setrecursionlimit(recursion_limit)



class RecordLinkBlocker(Blocker) :
    def tfIdfBlocks(self, data_1, data_2):
        '''Creates TF/IDF canopy of a given set of data'''
        
        if not self.tfidf_predicates:
            return
            
        tfidf_fields = set([])
        for predicate, field in self.tfidf_predicates :
            tfidf_fields.add(field)

        ii = tfidf.InvertedIndex(tfidf_fields)

        self.base_tokens = ii.unweightedIndex(data_1)
        target_tokens = ii.unweightedIndex(data_2)

        self.target_ii = {}

        for field, index in ii.inverted_indices.items() :
            weighted_vectors = mekano.WeightVectors(index)
            stop_words = tfidf.stopWords(index, 
                                          ii.stop_word_threshold)

            self.base_tokens[field] = tfidf.weightVectors(weighted_vectors,
                                                          self.base_tokens[field],
                                                          stop_words)

            print "hiya"

            self.target_ii = tfidf.removeStopWords(weighted_index)

            #targets = tfidf.weightVectors(weighted_vectors,
            #                              target_tokens[field],
            #                              stop_words)

            #self.target_ii[field] = tfidf.tokensToInvertedIndex(targets)


        self.createCanopies()



def blockTraining(training_pairs,
                  predicate_set,
                  eta=.1,
                  epsilon=.1,
                  pool=None,
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
                                      training_dupes + training_distinct,
                                      pool)

    else :
        coverage = DedupeCoverage(predicate_set,
                                  training_dupes + training_distinct,
                                  pool)

    coverage_threshold = eta * len(training_distinct)
    logging.info("coverage threshold: %s", coverage_threshold)

    # Only consider predicates that cover at least one duplicate pair
    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               training_dupes)
    predicate_set = dupe_coverage.keys()

    # We want to throw away the predicates that puts together too
    # many distinct pairs
    distinct_blocks = coverage.predicateBlocks(predicate_set,
                                               training_distinct)

    logging.info("Before removing liberal predicates, %s predicates",
                 len(predicate_set))

    for (pred, blocks) in distinct_blocks.iteritems():
        if any(len(block) >= coverage_threshold for block in blocks if block):
            predicate_set.remove(pred)

    logging.info("After removing liberal predicates, %s predicates",
                 len(predicate_set))

    distinct_coverage = coverage.predicateCoverage(predicate_set, 
                                                   training_distinct)

    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              distinct_coverage,
                                              epsilon,
                                              coverage)

    logging.info('Final predicate set:')
    for predicate in final_predicate_set :
        logging.info([(pred.__name__, field) for pred, field in predicate])

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
            logging.warning('Ran out of predicates')
            break

        final_predicate_set.append(best_predicate)
        predicate_set.remove(best_predicate)
        
        uncovered_dupes = uncovered_dupes - dupe_coverage[best_predicate]
        dupe_coverage = coverage.predicateCoverage(predicate_set,
                                                   uncovered_dupes)


        logging.debug([(pred.__name__, field)
                      for pred, field in best_predicate])
        logging.debug('cover: %(cover)f, found_dupes: %(found_dupes)d, '
                      'found_distinct: %(found_distinct)d, '
                      'uncovered dupes: %(uncovered)d',
                      {'cover' : best_cover,
                       'found_dupes' : best_dupes,
                       'found_distinct' : best_distinct,
                       'uncovered' : len(uncovered_dupes)
                       })




    return final_predicate_set




class Coverage(object) :
    def __init__(self, predicate_set, pairs, pool) :
        self.pool = pool

        self.stop_words = {}
        
        self.overlapping = defaultdict(set)
        self.blocks = defaultdict(lambda : defaultdict(set))

        basic_preds, tfidf_preds = predicateTypes(predicate_set)

        logging.info("Calculating coverage of simple predicates")
        self.simplePredicateOverlap(basic_preds, pairs)

        logging.info("Calculating coverage of tf-idf predicates")
        self.canopyOverlap(tfidf_preds, pairs)

        for predicate in predicate_set :
            covered_pairs = set.intersection(*(self.overlapping[basic_predicate]
                                               for basic_predicate
                                               in predicate))
            self.overlapping[predicate] = covered_pairs


    def simplePredicateOverlap(self,
                                basic_predicates,
                                pairs) :

        for basic_predicate in basic_predicates :
            (F, field) = basic_predicate        
            for pair in pairs :
                field_predicate_1 = F(pair[0][field])

                if field_predicate_1:
                    field_predicate_2 = F(pair[1][field])

                    if field_predicate_2 :
                        field_preds = set(field_predicate_2) & set(field_predicate_1)
                        if field_preds :
                            self.overlapping[basic_predicate].add(pair)

                        for field_pred in field_preds :
                            self.blocks[basic_predicate][field_pred].add(pair)


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

    def predicateBlocks(self,
                        predicate_set,
                        pairs) :

        predicate_blocks = {}
        blocks = defaultdict(lambda : defaultdict(set))

        pairs = set(pairs)

        
        for basic_predicate in self.blocks :
            for block_key, block_group in self.blocks[basic_predicate].iteritems() :
                block_group = pairs.intersection(block_group)
                if block_group :
                    blocks[basic_predicate][block_key] = block_group

        for predicate in predicate_set :
            block_groups = itertools.product(*(blocks[basic_predicate].values()
                                               for basic_predicate
                                               in predicate))

            block_groups = (set.intersection(*block_group)
                            for block_group in block_groups)
            predicate_blocks[predicate] = block_groups

        return predicate_blocks


class DedupeCoverage(Coverage) :
    def canopyOverlap(self,
                       tfidf_predicates,
                       record_pairs) :

        tfidf_fields = defaultdict(list)

        for threshold, field in tfidf_predicates :
            tfidf_fields[field].append(threshold)

        blocker = DedupeBlocker(pool=self.pool)
        blocker.tfidf_fields = tfidf_fields

        docs = list(set(itertools.chain(*record_pairs)))
        record_ids = dict(itertools.izip(docs, itertools.count()))
 
        for field in blocker.tfidf_fields :
            id_records = zip(itertools.count(), 
                             (record[field] for record in docs))

                                        
            # uniquify records

            blocker.tfIdfBlock(id_records, field, True)


        for canopy_id, canopy in blocker.canopies.items() :
            for record_1, record_2 in record_pairs :
                id_1 = record_ids[record_1]
                id_2 = record_ids[record_2]
                if canopy[id_1] == canopy[id_2]:
                    self.overlapping[canopy_id].add((record_1, record_2))
                    self.blocks[canopy_id][canopy[id_1]].add((record_1, record_2))
        self.stop_words = blocker.stop_words


class RecordLinkCoverage(Coverage) :

    def canopyOverlap(self,
                       tfidf_predicates,
                       record_pairs) :

        data_1 = set([])
        data_2 = set([])
        for record_1, record_2 in record_pairs :
            data_1.add(record_1)
            data_2.add(record_2)

        data_1 = list(itertools.izip(itertools.count(), 
                                     data_1))
        data_2 = list(itertools.izip(itertools.count(len(data_1)), 
                                     data_2))

        record_ids = dict((v, k) for k, v in data_1)
        record_ids.update(dict((v, k) for k, v in data_2))

        blocker = RecordLinkBlocker(pool=self.pool)
        blocker.tfidf_predicates = tfidf_predicates

        blocker.tfIdfBlocks(data_1, data_2)

        for (threshold, field) in blocker.tfidf_predicates:
            canopy = blocker.canopies[threshold.__name__ + field]
            for record_1, record_2 in record_pairs :
                id_1 = record_ids[record_1]
                id_2 = record_ids[record_2]
                if canopy[id_1] == canopy[id_2]:
                    self.overlapping[(threshold, field)].add((record_1, record_2))
                    self.blocks[(threshold, field)][canopy[id_1]].add((record_1, record_2))
    



def predicateTypes(predicates) :
    tfidf_predicates = set([])
    simple_predicates = set([])

    for predicate in predicates:
        for (pred, field) in predicate:
            if pred.__class__ is tfidf.TfidfPredicate:
                tfidf_predicates.add((pred, field))
            elif isinstance(pred, types.FunctionType):
                simple_predicates.add((pred, field))

    return simple_predicates, tfidf_predicates

def predicateGenerator(blocker_types, data_model) :
    predicate_set = []
    for record_type, predicate_functions in blocker_types.items() :
        fields = [field_name for field_name, details
                  in data_model['fields'].items()
                  if details['type'] == record_type]
        predicate_set.extend(list(itertools.product(predicate_functions, fields)))
    predicate_set = disjunctivePredicates(predicate_set)

    return predicate_set


def disjunctivePredicates(predicate_set):

    disjunctive_predicates = list(itertools.combinations(predicate_set, 2))

    # filter out disjunctive predicates that operate on same field

    disjunctive_predicates = [predicate for predicate in disjunctive_predicates 
                              if predicate[0][1] != predicate[1][1]]

    predicate_set = [(predicate, ) for predicate in predicate_set]
    predicate_set.extend(disjunctive_predicates)

    return predicate_set
