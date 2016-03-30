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

try :
    from functools import lru_cache
except ImportError :
    from backports.functools_lru_cache import lru_cache    

logger = logging.getLogger(__name__)

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

class BlockLearner(object) :
    def learn(self, matches, max_comparisons, recall) :
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        if len(self.blocker.predicates) <= 40 :
            compound_length = 3
        else :
            compound_length = 2

        self.blocker.indexAll({i : record
                               for i, record
                               in enumerate(self.unroll(matches))})
        
        dupe_cover = cover(self.blocker, matches, compound_length)

        comparison_count = self.comparisons(self.total_cover, compound_length)

        dupe_cover = {pred : pairs
                      for pred, pairs
                      in viewitems(dupe_cover)
                      if comparison_count[pred] < max_comparisons}

        if not dupe_cover : 
            raise ValueError(NO_PREDICATES_ERROR)

        uncoverable_dupes = set(matches) - set.union(*viewvalues(dupe_cover))

        epsilon = int((1.0 - recall) * len(matches))

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

    def comparisons(self, cover, compound_length) :
        CP = predicates.CompoundPredicate

        block_index = {}
        for predicate, blocks in viewitems(cover):
            block_index[predicate] = {}
            for block_id, blocks in viewitems(blocks) :
                for id in self._blocks(blocks) :
                    block_index[predicate].setdefault(id, set()).add(block_id)

        compounder = self.Compounder(cover, block_index)
        comparison_count = {}
        simple_predicates = sorted(cover, key=str)

        for i in range(2, compound_length+1) :
            for combo in itertools.combinations(simple_predicates, i) :
                comparison_count[CP(combo)] = sum(self.pairs(ids)
                                                  for ids in 
                                                  viewvalues(compounder(combo)))
        for pred in simple_predicates :
            comparison_count[pred] = sum(self.pairs(ids)
                                         for ids
                                         in viewvalues(cover[pred]))

        return comparison_count    

class Compounder(object) :
    def __init__(self, cover, block_index) :
        self.cover = cover
        self.block_index = block_index

    @lru_cache(2)
    def __call__(self, compound_predicate) :
        a, b = compound_predicate[:-1], compound_predicate[-1]

        if len(a) > 1 :
            cover_a = self(a)
        else :
            cover_a = self.cover[a[0]]

        return self.overlap(cover_a, b)

class DedupeCompounder(Compounder) :
    def overlap(self, cover_a, b) :
        b_index = self.block_index[b]
        b_ids = set(b_index)
        cover_b = self.cover[b]
        seen_blocks = set()
        seen_blocks_add = seen_blocks.add
        return {(x, y) : x_ids & cover_b[y]
                for x, x_ids in viewitems(cover_a)
                for id in x_ids & b_ids
                for y in b_index[id]
                if not ((x,y) in seen_blocks or seen_blocks_add((x,y)))}

class DedupeBlockLearner(BlockLearner) :
    Compounder = DedupeCompounder
    
    def __init__(self, predicates, sampled_records) :
        blocker = blocking.Blocker(predicates)
        blocker.indexAll(sampled_records)

        self.total_cover = self.coveredRecords(blocker, sampled_records)
        self.multiplier = sampled_records.original_length/len(sampled_records)

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches) : # pragma : no cover
        return set().union(*matches)

    @staticmethod
    def _blocks(blocks) : # pragma : no cover
        return blocks

    @staticmethod
    def coveredRecords(blocker, records) :
        CP = predicates.CompoundPredicate

        cover = {}
        block_index = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
            block_index[predicate] = {}
            for id, record in viewitems(records) :
                blocks = predicate(record)
                for block in blocks :
                    cover[predicate].setdefault(block, set()).add(id)

        return cover

    def pairs(self, ids) :
        N = len(ids) * self.multiplier
        return (N * (N - 1))/2

class RecordLinkCompounder(Compounder) :
    def overlap(self, cover_a, b) :
        b_index = self.block_index[b]
        first_b = set(b_index)
        cover_b = self.cover[b]
        seen_blocks = set()
        seen_blocks_add = seen_blocks.add
        return {(x, y) : (first & cover_b[y][0], second & cover_b[y][1])
                for x, (first, second) in viewitems(cover_a)
                for id in first & first_b
                for y in b_index[id]
                if not ((x,y) in seen_blocks or seen_blocks_add((x,y)))}
    
class RecordLinkBlockLearner(BlockLearner) :
    Compounder = RecordLinkCompounder
    
    def __init__(self, predicates, sampled_records_1, sampled_records_2) :
        blocker = blocking.Blocker(predicates)
        blocker.indexAll(sampled_records_2)

        self.total_cover = self.coveredRecords(blocker,
                                               sampled_records_1,
                                               sampled_records_2)

        self.multiplier_1 = sampled_records_1.original_length/len(sampled_records_1)
        self.multiplier_2 = sampled_records_2.original_length/len(sampled_records_2)

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches) : # pragma : no cover
        return {record_2 for _, record_2 in matches}

    @staticmethod
    def _blocks(blocks) : # pragma : no cover
        return blocks[0]
 
    @staticmethod
    def coveredRecords(blocker, records_1, records_2) :
        CP = predicates.CompoundPredicate

        cover = {}
        block_index = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
            block_index[predicate] = {}
            for id, record in viewitems(records_2) :
                blocks = predicate(record)
                for block in blocks :
                    cover[predicate].setdefault(block, (set(), set()))[1].add(id)

            current_blocks = set(cover[predicate])
            for id, record in viewitems(records_1) :
                blocks = set(predicate(record))
                for block in blocks & current_blocks :
                    cover[predicate][block][0].add(id)

        return cover

    def pairs(self, blocks) :
        A, B = blocks
        N = len(A) * self.multiplier_1
        M = len(B) * self.multiplier_2
        return N * M

    
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

def compound(cover, compound_length) :
    simple_predicates = sorted(cover, key=str)
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


        
        

OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"

NO_PREDICATES_ERROR = "No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data or increasing the `max_comparisons` argument to the train method"
