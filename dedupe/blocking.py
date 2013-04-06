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

    def tfIdfBlocks(self, data, df_index=None):
        '''Creates TF/IDF canopy of a given set of data'''
        
        if not self.tfidf_predicates:
            return
            
        tfidf_fields = set([])
        for predicate, field in self.tfidf_predicates :
            tfidf_fields.add(field)

        vectors = tfidf.invertIndex(data, tfidf_fields, df_index)
        inverted_index, token_vector, corpus_ids = vectors


        logging.info('creating TF/IDF canopies')

        num_thresholds = len(self.tfidf_predicates)

        for (i, (threshold, field)) in enumerate(self.tfidf_predicates, 1):
            logging.info('%(i)i/%(num_thresholds)i field %(threshold)2.2f %(field)s',
                         {'i': i, 
                          'num_thresholds': num_thresholds, 
                          'threshold': threshold, 
                          'field': field})

            canopy = tfidf.createCanopies(field, threshold, corpus_ids,
                                          token_vector, inverted_index)
            self.canopies[threshold.__name__ + field] = canopy


def blockTraining(training_pairs,
                  predicate_set,
                  fields,
                  eta=.1,
                  epsilon=.1):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    # Setup

    training_dupes = (training_pairs[1])[:]
    training_distinct = (training_pairs[0])[:]

    coverage = Coverage()

    basic_preds, tfidf_preds = predicateTypes(predicate_set)

    logging.info("Calculating coverage of simple predicates")
    coverage.simplePredicateOverlap(basic_preds,
                                    training_dupes + training_distinct)

    logging.info("Calculating coverage of tf-idf predicates")
    coverage.canopyOverlap(tfidf_preds,
                           training_dupes + training_distinct)


    coverage_threshold = eta * len(training_distinct)
    logging.info("coverage threshold: %s", coverage_threshold)

    # Only consider predicates that cover at least one duplicate pair
    found_dupes = coverage.predicateCoverage(predicate_set,
                                             training_dupes)
    predicate_set = found_dupes.keys()


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


    found_distinct = coverage.predicateCoverage(predicate_set, 
                                                training_distinct) 

    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              found_dupes,
                                              found_distinct,
                                              epsilon,
                                              coverage)

    logging.info('Final predicate set:')
    logging.info(final_predicate_set)

    if final_predicate_set:
        return final_predicate_set
    else:
        raise ValueError('No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data')


def findOptimumBlocking(training_dupes,
                        predicate_set,
                        found_dupes,
                        found_distinct,
                        epsilon,
                        coverage):

    # Greedily find the predicates that, at each step, covers the
    # most duplicates and covers the least distinct pairs, due to
    # Chvatal, 1979
    # print found_dupes

    final_predicate_set = []
    while len(training_dupes) > epsilon:

        optimum_cover = 0
        best_predicate = None
        for predicate in predicate_set:
            cover = len(found_dupes[predicate]) / (float(len(found_distinct[predicate])) + 0.5)
            if cover > optimum_cover:
                optimum_cover = cover
                best_predicate = predicate

                logging.debug(best_predicate)
                logging.debug('cover: %(cover)f, found_dupes: %(found_dupes)d, '
                              'found_distinct: %(found_distinct)d',
                              {'cover' : cover,
                               'found_dupes' : len(found_dupes[best_predicate]),
                               'found_distinct' : len(found_distinct[best_predicate])})



        if not best_predicate:
            logging.warning('Ran out of predicates')
            break

        logging.info(best_predicate)
        logging.info('cover: %(cover)f, found_dupes: %(found_dupes)d, '
                     'found_distinct: %(found_distinct)d',
                     {'cover' : cover,
                      'found_dupes' : len(found_dupes[best_predicate]),
                      'found_distinct' : len(found_distinct[best_predicate])
                      })

        [training_dupes.remove(pair) for pair in found_dupes[best_predicate]]

        predicate_set.remove(best_predicate)
        found_dupes  = coverage.predicateCoverage(predicate_set,
                                                  training_dupes)

        final_predicate_set.append(best_predicate)

    return final_predicate_set



class Coverage() :
    def __init__(self) :
        self.overlapping = defaultdict(set)
        self.blocks = defaultdict(lambda : defaultdict(set))

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
                       record_pairs) :

        # uniquify records
        docs = list(set(itertools.chain(*record_pairs)))

        self_identified = itertools.izip(docs, docs)

        blocker = Blocker()
        blocker.tfidf_predicates = tfidf_predicates
        blocker.tfIdfBlocks(self_identified)

        for (threshold, field) in blocker.tfidf_predicates:
            canopy = blocker.canopies[threshold.__name__ + field]
            for record_1, record_2 in record_pairs :
                if canopy[record_1] == canopy[record_2]:
                    self.overlapping[(threshold, field)].add((record_1, record_2))
                    self.blocks[(threshold, field)][canopy[record_1]].add((record_1, record_2))


    def predicateCoverage(self,
                          predicate_set,
                          pairs) :
    
        overlapping = defaultdict(set)
        coverage = defaultdict(list)

        pairs = set(pairs)

        for basic_predicate, covered_pairs in self.overlapping.iteritems() :
            overlapping[basic_predicate] = pairs.intersection(covered_pairs)

        for predicate in predicate_set :
            covered_pairs = set.intersection(*(overlapping[basic_predicate]
                                               for basic_predicate
                                               in predicate))
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


