#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data
from __future__ import division
from future.utils import viewitems, viewkeys, viewvalues

from collections import defaultdict
import itertools
import functools
from . import blocking, predicates, core, index, backport
import numpy
import logging
import random

logger = logging.getLogger(__name__)

def comparisons(block_count) :
    comparison_count = {}
    for predicate, blocks in viewitems(block_count) :
        count = 0
        for covered_ids in viewvalues(blocks) :
            N = len(covered_ids)
            count += (N * N-1)/2
        comparison_count[predicate] = count

    return comparison_count

def findUncertainPairs(field_distances, classifier, bias=0.5):
    """
    Given a set of field distances and a data model return the
    indices of the record pairs in order of uncertainty. For example,
    the first indices corresponds to the record pair where we have the
    least certainty whether the pair are duplicates or distinct.
    """

    probability = classifier.predict_proba(field_distances)[:,-1]

    p_max = (1 - bias)
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
    def __init__(self, candidates, data_model, num_processes) :

        self.candidates = candidates

        pool = backport.Pool(num_processes)
        self.field_distances = numpy.concatenate(
            pool.map(data_model.distances, 
                     chunker(candidates, 100),
                     2))
        
        pool.terminate()

        self.seen_indices = set()

    def uncertainPairs(self, classifier, dupe_proportion) :
        uncertain_indices = findUncertainPairs(self.field_distances,
                                               classifier,
                                               dupe_proportion)

        for uncertain_index in uncertain_indices:
            if uncertain_index not in self.seen_indices:
                self.seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [self.candidates[uncertain_index]]

        return uncertain_pairs

def trainingData(training_pairs, record_ids) :

    record_pairs = set()
    tuple_pairs = set()
    for pair in training_pairs :
        record_pairs.add(tuple([(record_ids[record], record) 
                                for record in pair]))
        tuple_pairs.add(tuple([record_ids[record] 
                               for record in pair]))
    return record_pairs, tuple_pairs
    

def blockTraining(pairs,
                  predicate_set,
                  records,
                  max_comparisons,
                  recall,
                  matching = "Dedupe",
                  num_records=None):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    blocker = blocking.Blocker(predicate_set)
    prepare_index(blocker, pairs, matching)

    if len(pairs['match']) < 50 :
        compound_length = 2
    else :
        compound_length = 2

    dupe_cover = cover(blocker, pairs['match'], compound_length)

    total_cover = coveredRecords(blocker, records, 2)
    comparison_count = comparisons(total_cover)
        
    del total_cover

    import pdb
    pdb.set_trace()
    
    dupe_cover = {pred : pairs
                  for pred, pairs
                  in viewitems(dupe_cover)
                  if comparison_count[pred] < max_comparisons}

    if not dupe_cover : 
        raise ValueError(NO_PREDICATES_ERROR)

    uncoverable_dupes = set(pairs['match']) - set.union(*viewvalues(dupe_cover))

    epsilon = int((1.0 - recall) * len(pairs['match']))
    
    if len(uncoverable_dupes) > epsilon :
        logger.warning(OUT_OF_PREDICATES_WARNING)
        logger.debug(uncoverable_dupes)
        epsilon = 0
    else :
        epsilon -= len(uncoverable_dupes)

    chvatal_set = greedy(dupe_cover.copy(), comparison_count, epsilon)

    dupe_cover = {pred : dupe_cover[pred] for pred in chvatal_set}
        
    final_predicates = tuple(dominating(dupe_cover))

    logger.info('Final predicate set:')
    for predicate in final_predicates :
        logger.info(predicate)

    return final_predicates

def greedy(dupe_cover, comparison_count, epsilon):

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

    uncovered_dupes = set.union(*dupe_cover.values())
    final_predicates = set()

    while len(uncovered_dupes) > epsilon and dupe_cover :
        cost = lambda p : comparison_count[p]/len(dupe_cover[p])
        
        best_predicate = min(dupe_cover, key = cost)
        final_predicates.add(best_predicate)

        print(cost(best_predicate))

        covered = dupe_cover.pop(best_predicate)        
        uncovered_dupes = uncovered_dupes - covered
        remaining_cover(dupe_cover, covered)

        logger.debug(best_predicate)
        logger.debug('uncovered dupes: %(uncovered)d',
                     {'uncovered' : len(uncovered_dupes)})

    return final_predicates

def dominating(dupe_cover) :

    uncovered_dupes = set.union(*dupe_cover.values())
    final_predicates = set()

    while uncovered_dupes :
        score = lambda p : len(dupe_cover[p])

        best_predicate = max(dupe_cover, key=score)
        final_predicates.add(best_predicate)

        covered = dupe_cover.pop(best_predicate)
        uncovered_dupes = uncovered_dupes - covered

        remaining_cover(dupe_cover, covered)

    return final_predicates

def cover(blocker, pairs, compound_length) :
    cover = coveredPairs(blocker.predicates, pairs)
    cover = compound(cover, compound_length)
    remaining_cover(cover)
    return cover

def coveredPairs(predicates, pairs) :
    cover = {}
    pairs = sorted(pairs)
        
    for predicate in predicates :
        rec_1 = None
        for pair in pairs :
            record_1, record_2 = pair
            if record_1 != rec_1 :
                blocks_1 = set(predicate(record_1))
                rec_1 = record_1

            if blocks_1 :
                blocks_2 = predicate(record_2)
                field_preds = blocks_1 & set(blocks_2)
                if field_preds :
                    cover.setdefault(predicate, set()).add(pair)

    return cover

def coveredRecords(blocker, records, compound_length) :
    CP = predicates.CompoundPredicate

    cover = defaultdict(lambda : defaultdict(set))
    block_index = defaultdict(lambda : defaultdict(set))

    for predicate in blocker.predicates :
        for id, record in enumerate(viewvalues(records)) :
            blocks = predicate(record)
            for block in blocks :
                cover[predicate][block].add(id)
                block_index[predicate][id].add(block)

    i = 0
    for a, b in itertools.combinations(sorted(cover), 2) :
        i += 1
        print(i)
        ab = CP((a,b))
        for x, ids in viewitems(cover[a]) :
            seen_blocks = set()
            for id in ids :
                for y in block_index[b][id] - seen_blocks :
                    cover[ab][x,y] = ids & cover[b][y]
                    seen_blocks.add(y)

    return cover
                    
        

def compound(cover, compound_length) :
    simple_predicates = sorted(cover)
    CP = predicates.CompoundPredicate

    for i in range(2, compound_length+1) :
        compound_predicates = itertools.combinations(simple_predicates, i)
                                                             
        for compound_predicate in compound_predicates :
            a, b = compound_predicate[:-1], compound_predicate[-1]
            if len(a) == 1 :
                a = a[0]

            cover[CP(compound_predicate)] = cover[a] & cover[b]

    return cover

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def remaining_cover(coverage, covered=set()) :
    null_covers = []
    for predicate, uncovered in viewitems(coverage) :
        uncovered -= covered
        if not uncovered :
            null_covers.append(predicate)

    for predicate in null_covers :
        del coverage[predicate]

def prepare_index(blocker, pairs, matching) :
    if matching == "RecordLink" :
        unroll = lambda p : {record_2 for _, record_2 in p}
    else :
        unroll = lambda p : set().union(*p)

    records = unroll(itertools.chain.from_iterable(viewvalues(pairs)))
    
    for field, indices in blocker.index_fields.items() :
        record_fields = [record[field] 
                         for record 
                         in records
                         if record[field]]
        blocker.index(sorted(set(record_fields)), field)

OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the ppc argument to the train method"

NO_PREDICATES_ERROR = "No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data or increasing the ppc argument to the train method"
