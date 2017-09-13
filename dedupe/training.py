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
    def learn(self, matches, recall) :
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        compound_length = 2

        self.blocker.indexAll({i : record
                               for i, record
                               in enumerate(self.unroll(matches))})

        dupe_cover = cover(self.blocker, matches,
                           self.total_cover, compound_length)

        self.blocker.resetIndices()

        comparison_count = self.comparisons(dupe_cover, compound_length)

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

        searcher = BranchBound(dupe_cover, comparison_count, epsilon, 2500)
        final_predicates = searcher.search(dupe_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def comparisons(self, match_cover, compound_length) :
        compounder = Compounder(self.total_cover)
        comparison_count = {}

        for pred in sorted(match_cover, key=lambda x: len(match_cover[x])):
            if len(pred) > 1:
                comparison_count[pred] = self.estimate(compounder(pred))
            else:
                comparison_count[pred] = self.estimate(self.total_cover[pred])

        return comparison_count    

class Compounder(object) :
    def __init__(self, cover):
        self.cover = cover

    def __call__(self, compound_predicate) :
        return set.intersection(*(self.cover[pred]
                                  for pred in compound_predicate))

class DedupeBlockLearner(BlockLearner) :
    
    def __init__(self, predicates, sampled_records) :
        self.pair_id = core.Enumerator()

        blocker = blocking.Blocker(predicates)
        blocker.indexAll(sampled_records)

        self.total_cover = self.coveredPairs(blocker, sampled_records)
        self.multiplier = sampled_records.original_length/len(sampled_records)

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches) : # pragma: no cover
        return unique((record for pair in matches for record in pair))

    def coveredPairs(self, blocker, records) :
        cover = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
            for id, record in viewitems(records) :
                blocks = predicate(record)
                for block in blocks :
                    cover[predicate].setdefault(block, set()).add(id)

        for predicate, blocks in cover.items():
            pairs = set()
            for block in blocks.values():
                for pair in itertools.combinations(sorted(block), 2):
                    pairs.add(self.pair_id[pair])
            cover[predicate] = pairs

        return cover

    def estimate(self, blocks):
        return len(blocks) * self.multiplier * self.multiplier

class RecordLinkBlockLearner(BlockLearner) :
    
    def __init__(self, predicates, sampled_records_1, sampled_records_2) :
        self.pair_id = core.Enumerator()
        
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

    def coveredPairs(self, blocker, records_1, records_2) :
        cover = {}

        for predicate in blocker.predicates :
            cover[predicate] = {}
            for id, record in viewitems(records_2) :
                blocks = predicate(record, target=True)
                for block in blocks :
                    cover[predicate].setdefault(block, (set(), set()))[1].add(id)

            current_blocks = set(cover[predicate])
            for id, record in viewitems(records_1) :
                blocks = set(predicate(record))
                for block in blocks & current_blocks :
                    cover[predicate][block][0].add(id)

        for predicate, blocks in cover.items():
            pairs = set()
            for A, B in blocks.values():
                for pair in itertools.product(A, B):
                    pairs.add(self.pair_id[pair])
            cover[predicate] = pairs

        return cover

    def estimate(self, blocks):
        return len(blocks) * self.multiplier_1 * self.multiplier_2

    
class BranchBound(object) :
    def __init__(self, original_cover, comparison_count, epsilon, max_calls) :
        self.original_cover = original_cover.copy()
        self.calls = max_calls
        self.comparisons = comparison_count

        self.target = (len(set.union(*original_cover.values())) -
                       epsilon)

        self.cheapest = tuple(original_cover.keys())
        self.cheapest_score = float('inf')

    def search(self, candidates, partial=()) :
        if self.calls <= 0 :
            return self.cheapest

        self.calls -= 1

        covered = self.covered(partial)
        score = self.score(partial)

        if covered >= self.target:
            if score < self.cheapest_score :
                self.cheapest = partial
                self.cheapest_score = score
                print(self.calls, self.cheapest_score)


        else:
            window = self.cheapest_score - score

            candidates = {p : cover
                          for p, cover in viewitems(candidates)
                          if self.comparisons[p] < window}

            reachable = self.reachable(candidates) + covered

            if candidates and reachable >= self.target:
                best = max(candidates, key=lambda p: len(candidates[p]))

                remaining = remaining_cover(candidates,
                                            candidates[best])
                self.search(remaining, partial + (best,))
                del remaining

                reduced = self.dominates(candidates, best)
                self.search(reduced, partial)

        return self.cheapest

    def score(self, partial):
        return sum(self.comparisons[p] for p in partial)

    def covered(self, partial):
        if partial:
            return len(set.union(*(self.original_cover[p]
                                   for p in partial)))
        else:
            return 0

    def reachable(self, dupe_cover):
        if dupe_cover:
            return len(set.union(*dupe_cover.values()))
        else:
            return 0

    def dominates(self, coverage, dominator):
        dominant_cost = self.comparisons[dominator]
        dominant_cover = coverage[dominator]

        for pred, cover in viewitems(coverage.copy()):
             if (dominant_cost <= self.comparisons[pred] and
                 dominant_cover >= cover):
                 del coverage[pred]

        return coverage


def cover(blocker, pairs, total_cover, compound_length): # pragma: no cover
    cover = coveredPairs(blocker.predicates, pairs)
    cover = dominators(cover, total_cover)
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

            if a in cover:
                compound_cover = cover[a] & cover[b]
                if compound_cover:
                    cover[CP(compound_predicate)] = compound_cover

    return cover

def remaining_cover(coverage, covered=set()):
    remaining = {}
    for predicate, uncovered in viewitems(coverage):
        still_uncovered = uncovered - covered
        if still_uncovered:
            if still_uncovered == uncovered:
                remaining[predicate] = uncovered
            else:
                remaining[predicate] = still_uncovered

    return remaining

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def dominators(match_cover, total_cover):
    for a, b in itertools.combinations(list(match_cover), 2):
        if a not in match_cover or b not in match_cover:
            continue
        cover_a, cover_b = match_cover[a], match_cover[b]
        if len(cover_b) > len(cover_a):
            a, b = b, a
            cover_a, cover_b = cover_b, cover_a
        if cover_a >= cover_b and total_cover[a] <= total_cover[b]:
            del match_cover[b]

    return match_cover


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"
