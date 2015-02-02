#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

from collections import defaultdict
import itertools
from itertools import combinations, islice
import blocking
import predicates
import core
import numpy
import logging
import random
import index

logger = logging.getLogger(__name__)

def findUncertainPairs(field_distances, data_model, bias=0.5):
    """
    Given a set of field distances and a data model return the
    indices of the record pairs in order of uncertainty. For example,
    the first indices corresponds to the record pair where we have the
    least certainty whether the pair are duplicates or distinct.
    """

    probability = core.scorePairs(field_distances, data_model)

    p_max = (1.0 - bias)
    logger.info(p_max)

    informativity = numpy.copy(probability)
    informativity[probability < p_max] /= p_max
    informativity[probability >= p_max] = (1 - probability[probability >= p_max])/(1-p_max)


    return numpy.argsort(-informativity)


class ActiveLearning(object) :
    """
    Ask the user to label the record pair we are most uncertain of. Train the
    data model, and update our uncertainty. Repeat until user tells us she is
    finished.
    """
    def __init__(self, candidates, data_model) :

        self.candidates = candidates
        self.field_distances = core.fieldDistances(candidates, data_model)
        self.seen_indices = set()

    def uncertainPairs(self, data_model, dupe_ratio) :
        uncertain_indices = findUncertainPairs(self.field_distances,
                                               data_model,
                                               dupe_ratio)

        for uncertain_index in uncertain_indices:
            if uncertain_index not in self.seen_indices:
                self.seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [self.candidates[uncertain_index]]

        return uncertain_pairs


def semiSupervisedNonDuplicates(data_sample,
                                data_model,
                                nonduplicate_confidence_threshold=.9,
                                sample_size=2000):

    confidence = 1 - nonduplicate_confidence_threshold

    def distinctPairs() :
        data_slice = data_sample[0:sample_size]
        pair_distance = core.fieldDistances(data_slice, data_model)
        scores = core.scorePairs(pair_distance, data_model)

        sample_n = 0
        for score, pair in zip(scores, data_sample) :
            if score < confidence :
                yield pair
                sample_n += 1

        if sample_n < sample_size and len(data_sample) > sample_size :
            for pair in data_sample[sample_size:] :
                pair_distance = core.fieldDistances([pair], data_model)
                score = core.scorePairs(pair_distance, data_model)
                
                if score < confidence :
                    yield (pair)

    return islice(distinctPairs(), 0, sample_size)

def trainingData(training_pairs, record_ids) :
    record_pairs = set([])
    tuple_pairs = set([])
    for pair in training_pairs :
        record_pairs.add(tuple([(record_ids[record], record) 
                                for record in pair]))
        tuple_pairs.add(tuple([record_ids[record] 
                               for record in pair]))
    return record_pairs, tuple_pairs
    

def blockTraining(training_pairs,
                  predicate_set,
                  eta=.1,
                  epsilon=.1,
                  matching = "Dedupe"):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    if matching == "RecordLink" :
        Coverage = RecordLinkCoverage
    else :
        Coverage = DedupeCoverage

    # Setup

    record_ids = {}

    i = 0
    for record_set in training_pairs.values() :
        for pair in record_set :
            for record in pair :
                record_ids[record] = i
                i += 1

    dupe_pairs, training_dupes = trainingData(training_pairs['match'],
                                              record_ids)

    distinct_pairs, training_distinct = trainingData(training_pairs['distinct'],
                                                     record_ids)
    
    coverage = Coverage(predicate_set,
                        dupe_pairs | distinct_pairs)

    predicate_set = coverage.overlap.keys()
    
    # Only consider predicates that cover at least one duplicate pair
    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               training_dupes)
    predicate_set = dupe_coverage.keys()

    # Throw away the predicates that cover too many distinct pairs
    coverage_threshold = eta * len(training_distinct)
    logger.info("coverage threshold: %s", coverage_threshold)

    distinct_coverage = coverage.predicateCoverage(predicate_set,
                                                   training_distinct)

    for pred, pairs in distinct_coverage.items() :
        if len(pairs) > coverage_threshold :
            predicate_set.remove(pred)

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

    def coveredBy(self, predicates, pairs) :
        self.overlap = defaultdict(set)

        for pair in pairs :
            (record_1_id, record_1), (record_2_id, record_2) = pair
            for predicate in predicates :
                blocks_1 = predicate(record_1_id, record_1)
                blocks_2 = predicate(record_2_id, record_2)
                field_preds = set(blocks_1) & set(blocks_2)
                if field_preds :
                    rec_pair = record_1_id, record_2_id
                    self.overlap[predicate].add(rec_pair)


    def predicateCoverage(self,
                          predicate_set,
                          pairs) :

        coverage = defaultdict(set)
        pairs = set(pairs)

        for predicate in predicate_set :
            covered_pairs = pairs.intersection(self.overlap[predicate])
            if covered_pairs :
                coverage[predicate] = covered_pairs

        return coverage

    def compoundPredicates(self) :
        intersection = set.intersection
        product = itertools.product

        compound_predicates = itertools.combinations(self.overlap, 2)

        for compound_predicate in compound_predicates :
            compound_predicate = predicates.CompoundPredicate(compound_predicate)
            predicate_1, predicate_2 = compound_predicate
        
            self.overlap[compound_predicate] =\
                intersection(self.overlap[predicate_1],
                             self.overlap[predicate_2])
            
            i = 0


class DedupeCoverage(Coverage) :
    def __init__(self, predicate_set, pairs) :

        records = set(itertools.chain(*pairs))

        blocker = blocking.DedupeBlocker(predicate_set)

        for field in blocker.tfidf_fields :
            field_records = [(record_id, record[field]) 
                             for record_id, record in records]
            stop_words = stopWords(field_records)
            blocker.stop_words[field].update(stop_words)
            blocker.tfIdfBlock(field_records, field)

        self.stop_words = blocker.stop_words
        self.coveredBy(blocker.predicates, pairs)
        self.compoundPredicates()
        blocker._resetCanopies()

class RecordLinkCoverage(Coverage) :
    def __init__(self, predicate_set, pairs) :

        records_1 = set([])
        records_2 = set([])

        for record_1, record_2 in pairs :
            records_1.add(record_1)
            records_2.add(record_2)

        blocker = blocking.RecordLinkBlocker(predicate_set)

        for field in blocker.tfidf_fields :
            field_records = [(record_id, record[field]) 
                             for record_id, record in records_2]
            stop_words = stopWords(field_records)
            blocker.stop_words[field].update(stop_words)
            blocker.tfIdfIndex(field_records, field)

            search_records = [(record_id, record[field]) 
                              for record_id, record in records_1]

        canopies = defaultdict(lambda:defaultdict(set))

        for field in blocker.tfidf_fields :
            for source_id, record in search_records :
                for predicate in blocker.tfidf_fields[field] :
                    candidates = predicate(source_id, {field : record})
                    for target_id in candidates :
                        canopies[predicate][source_id].add(target_id)

        for field in blocker.tfidf_fields :
            for predicate in blocker.tfidf_fields[field] :
                predicate.canopy.update(canopies[predicate])
                del predicate.index
            
        self.stop_words = blocker.stop_words
        self.coveredBy(blocker.predicates, pairs)
        self.compoundPredicates()
        blocker._resetCanopies()
        

def stopWords(data) :
    tf_index = index.CanopyIndex([])

    for i, (_, doc) in enumerate(data, 1) :
        tf_index.index_doc(i, doc)

    doc_freq = [(len(tf_index.index._wordinfo[wid]), word) 
                for word, wid in tf_index.lexicon.items()]

    doc_freq.sort(reverse=True)

    N = float(tf_index.index.documentCount())
    threshold = int(max(1000, N * 0.05))

    stop_words = set([])

    for frequency, word in doc_freq :
        if frequency > threshold :
            stop_words.add(word)
        else :
            break

    return stop_words

    



