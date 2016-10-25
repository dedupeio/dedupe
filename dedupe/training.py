#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data
from __future__ import division
from future.utils import viewitems, viewvalues

import itertools
from . import blocking, predicates, core
import numpy
import logging

logger = logging.getLogger(__name__)

class BlockLearner(object) :
    def learn(self, matches, max_comparisons, recall) :
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
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

        coverable_dupes = set.union(*viewvalues(dupe_cover))
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]

        epsilon = int((1.0 - recall) * len(matches))

        if len(uncoverable_dupes) > epsilon :
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            epsilon = 0
        else :
            epsilon -= len(uncoverable_dupes)

        searcher = BranchBound(dupe_cover, comparison_count, epsilon, 5000)
        final_predicates = searcher.search(dupe_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def comparisons(self, cover, compound_length) :
        CP = predicates.CompoundPredicate

        block_index = {}
        for predicate, blocks in viewitems(cover):
            block_index[predicate] = {}
            for block_id, blocks in viewitems(blocks) :
                for record_id in self._blocks(blocks) :
                    block_index[predicate].setdefault(record_id,
                                                      set()).add(block_id)

        compounder = self.Compounder(cover, block_index)
        comparison_count = {}
        simple_predicates = sorted(cover, key=str)

        for i in range(2, compound_length+1) :
            for combo in itertools.combinations(simple_predicates, i) :
                comparison_count[CP(combo)] = self.estimate(compounder(combo))
        for pred in simple_predicates :
            comparison_count[pred] = self.estimate(viewvalues(cover[pred]))

        return comparison_count    

class Compounder(object) :
    def __init__(self, cover, block_index) :
        self.cover = cover
        self.block_index = block_index
        self._cached_predicate = None
        self._cached_blocks = None
        
    def __call__(self, compound_predicate) :
        a, b = compound_predicate[:-1], compound_predicate[-1]

        if len(a) > 1 :
            if a == self._cached_predicate :
                a_blocks = self._cached_blocks
            else :
                a_blocks = self._cached_blocks = list(self(a))
                self._cached_predicate = a
        else :
            a_blocks = viewvalues(self.cover[a[0]])

        return self.overlap(a_blocks, b)

class DedupeCompounder(Compounder) :
    def overlap(self, a_blocks, b) :
        b_index = self.block_index[b]
        b_index_get = b_index.get
        cover_b = self.cover[b]
        null_set = set()
        for x_ids in a_blocks :
            seen_y = set()
            for record_id in x_ids :
                b_blocks = b_index_get(record_id, null_set)
                for y in b_blocks :
                    if y not in seen_y :
                        yield x_ids & cover_b[y]
                seen_y |= b_blocks


class DedupeBlockLearner(BlockLearner) :
    Compounder = DedupeCompounder
    
    def __init__(self, predicates, sampled_records) :
        blocker = blocking.Blocker(predicates)
        blocker.indexAll(sampled_records)

        self.total_cover = self.coveredRecords(blocker, sampled_records)
        self.multiplier = sampled_records.original_length/len(sampled_records)

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches) : # pragma: no cover
        return unique((record for pair in matches for record in pair))

    @staticmethod
    def _blocks(blocks) : # pragma: no cover
        return blocks

    @staticmethod
    def coveredRecords(blocker, records) :
        cover = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
            for id, record in viewitems(records) :
                blocks = predicate(record)
                for block in blocks :
                    cover[predicate].setdefault(block, set()).add(id)

        return cover

    def estimate(self, blocks):
        lengths = numpy.fromiter((len(ids) for ids in blocks), int)
        return numpy.sum((lengths * self.multiplier) *
                         (lengths * self.multiplier - 1) /
                         2)

class RecordLinkCompounder(Compounder) :
    def overlap(self, a_blocks, b) :
        b_index = self.block_index[b]
        b_index_get = b_index.get
        cover_b = self.cover[b]
        null_set = set()
        for first, second in a_blocks:
            seen_y = set()
            for record_id in first:
                b_blocks = b_index_get(record_id, null_set)
                for y in b_blocks:
                    if y not in seen_y :
                        yield first & cover_b[y][0], second & cover_b[y][1]
                    seen_y |= b_blocks
    
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
    def unroll(matches) : # pragma: no cover
        return unique((record_2 for _, record_2 in matches))

    @staticmethod
    def _blocks(blocks) : # pragma: no cover
        return blocks[0]
 
    @staticmethod
    def coveredRecords(blocker, records_1, records_2) :
        cover = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
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

    def estimate(self, blocks):
        A, B = core.iunzip(blocks, 2)
        lengths_A = numpy.fromiter((len(ids) for ids in A), float)
        lengths_B = numpy.fromiter((len(ids) for ids in B), float)
        return numpy.sum((lengths_A * self.multiplier_1) *
                         (lengths_B * self.multiplier_2))


class BranchBound(object) :
    def __init__(self, original_cover, comparison_count, epsilon, max_calls) :
        self.dupes_to_cover = set.union(*original_cover.values())
        self.original_cover = original_cover.copy()
        self.calls = max_calls
        self.comparisons = comparison_count
        self.epsilon = epsilon

        self.cheapest = tuple(original_cover.keys())
        self.cheapest_score = float('inf')

    def search(self, dupe_cover, partial=()) :
        if self.calls <= 0 :
            return self.cheapest

        self.calls -= 1

        uncovered_dupes = self.dupes_to_cover.difference(*(self.original_cover[pred]
                                                           for pred in partial))

        if len(uncovered_dupes) <= self.epsilon :
            partial_score = self.score(partial)
            if partial_score < self.cheapest_score :
                self.cheapest = partial
                self.cheapest_score = partial_score

        elif dupe_cover and (self.lower_bound(partial, dupe_cover) <= self.cheapest_score):
            cost = lambda p : self.comparisons[p]/len(dupe_cover[p])
            best_predicate = min(dupe_cover, key=cost)

            remaining = remaining_cover(dupe_cover, dupe_cover[best_predicate])

            self.search(remaining, partial + (best_predicate,))

            reduced = self.dominates(dupe_cover, best_predicate)

            uncoverable_dupes = uncovered_dupes.difference(*reduced.values())
            if len(uncoverable_dupes) <= self.epsilon:
                self.search(reduced, partial)

        return self.cheapest

    def score(self, partial) :
        return sum(self.comparisons[p] for p in partial)

    def lower_bound(self, partial, dupe_cover) :
        return self.score(partial) + min(self.comparisons[p] for p in dupe_cover)

    def dominates(self, coverage, dominator):
        remaining = coverage.copy()

        dominant_cost = self.comparisons[dominator]
        dominant_cover = coverage[dominator]

        for pred, cover in viewitems(coverage):
             if (dominant_cost <= self.comparisons[pred] and
                 dominant_cover >= cover):
                 del remaining[pred]

        return remaining


def cover(blocker, pairs, compound_length) :
    cover = coveredPairs(blocker.predicates, pairs)
    cover = compound(cover, compound_length)
    cover = remaining_cover(cover)
    return cover

def coveredPairs(predicates, pairs) :
    cover = {}
        
    for predicate in predicates :
        cover[predicate] = {i for i, (record_1, record_2)
                            in enumerate(pairs)
                            if (set(predicate(record_1)) &
                                set(predicate(record_2)))}

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

def remaining_cover(coverage, covered=set()):
    remaining = {}
    for predicate, uncovered in viewitems(coverage):
        still_uncovered = uncovered - covered
        if still_uncovered:
            remaining[predicate] = still_uncovered

    return remaining

                            

def unique(seq):
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"

NO_PREDICATES_ERROR = "No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data or increasing the `max_comparisons` argument to the train method"
