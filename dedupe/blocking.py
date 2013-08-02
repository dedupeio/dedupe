#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import types
import logging

import dedupe.tfidf as tfidf

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, predicates = None):

        if predicates is None :
            self.simple_predicates = set([])
            self.tfidf_predicates = set([])

        else :
            predicate_types = predicateTypes(predicates)
            self.simple_predicates, self.tfidf_predicates = predicate_types

        self.predicates = predicates
        self.canopies = {}

    def __call__(self, instance):
        if self.tfidf_predicates and not self.canopies :
            raise ValueError("No canopies defined, but tf-idf predicate "
                             "learned. Did you run the tfIdfBlocks method "
                             "of the blocker?")

        (record_id, record) = instance

        record_keys = []
        for predicate in self.predicates:
            predicate_keys = []
            for (F, field) in predicate:
                pred_id = F.__name__ + field
                if isinstance(F, types.FunctionType):
                    record_field = record[field]
                    block_keys = [str(key) + pred_id for key in F(record_field)]
                    predicate_keys.append(block_keys)
                elif F.__class__ is tfidf.TfidfPredicate:
                    center = self.canopies[pred_id][record_id]
                    if center is not None:
                        key = str(center) + pred_id
                        predicate_keys.append((key, ))
                    else:
                        continue

            record_keys.extend(itertools.product(*predicate_keys))

        return set([str(key) for key in record_keys])

    def tfIdfBlocks(self, data, constrained_matching=False):
        '''Creates TF/IDF canopy of a given set of data'''
        
        if not self.tfidf_predicates:
            return
            
        tfidf_fields = set([])
        for predicate, field in self.tfidf_predicates :
            tfidf_fields.add(field)

        vectors = tfidf.invertIndex(data, tfidf_fields, constrained_matching)
        inverted_index, token_vector, corpus_ids = vectors


        logging.info('creating TF/IDF canopies')

        num_thresholds = len(self.tfidf_predicates)

        for (i, (threshold, field)) in enumerate(self.tfidf_predicates, 1):
            logging.info('%(i)i/%(num_thresholds)i field %(threshold)2.2f %(field)s',
                         {'i': i, 
                          'num_thresholds': num_thresholds, 
                          'threshold': threshold, 
                          'field': field})

            canopy = tfidf.createCanopies(field, data, threshold, corpus_ids,
                                          token_vector, inverted_index, constrained_matching)
            self.canopies[threshold.__name__ + field] = canopy


def blockTraining(training_pairs,
                  predicate_set,
                  constrained_matching=False,
                  eta=.1,
                  epsilon=.1):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    # Setup

    training_dupes = (training_pairs[1])[:]
    training_distinct = (training_pairs[0])[:]

    coverage = Coverage(predicate_set,
                        training_dupes + training_distinct,
                        constrained_matching)

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
        return final_predicate_set
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



class Coverage() :
    def __init__(self, predicate_set, pairs, constrained_matching=False) :
        self.overlapping = defaultdict(set)
        self.blocks = defaultdict(lambda : defaultdict(set))

        basic_preds, tfidf_preds = predicateTypes(predicate_set)

        logging.info("Calculating coverage of simple predicates")
        self.simplePredicateOverlap(basic_preds, pairs)

        logging.info("Calculating coverage of tf-idf predicates")
        self.canopyOverlap(tfidf_preds, pairs, constrained_matching)

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

    def canopyOverlap(self,
                       tfidf_predicates,
                       record_pairs,
                       constrained_matching=False) :

        # uniquify records
        docs = list(set(itertools.chain(*record_pairs)))

        self_identified = itertools.izip(docs, docs)

        blocker = Blocker()
        blocker.tfidf_predicates = tfidf_predicates
        blocker.tfIdfBlocks(self_identified,constrained_matching)

        for (threshold, field) in blocker.tfidf_predicates:
            canopy = blocker.canopies[threshold.__name__ + field]
            for record_1, record_2 in record_pairs :
                if canopy[record_1] == canopy[record_2]:
                    self.overlapping[(threshold, field)].add((record_1, record_2))
                    self.blocks[(threshold, field)][canopy[record_1]].add((record_1, record_2))


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


