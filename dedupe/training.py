#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data
from __future__ import division
from future.utils import viewitems, viewvalues, viewkeys

import itertools
import logging
import collections
import functools

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from . import blocking, predicates, core

logger = logging.getLogger(__name__)


class BlockLearner(object):
    def learn(self, matches, recall):
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        comparison_count = self.comparison_count

        dupe_cover = Cover(self.blocker.predicates, matches)
        dupe_cover.compound(2)
        dupe_cover.intersection_update(comparison_count)

        dupe_cover.dominators(cost=comparison_count)

        coverable_dupes = set.union(*viewvalues(dupe_cover))
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]

        epsilon = int((1.0 - recall) * len(matches))

        if len(uncoverable_dupes) > epsilon:
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            epsilon = 0
        else:
            epsilon -= len(uncoverable_dupes)

        for pred in dupe_cover:
            pred.count = comparison_count[pred]

        searcher = BranchBound(len(coverable_dupes) - epsilon, 2500)
        final_predicates = searcher.search(dupe_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def compound(self, simple_predicates, compound_length):
        simple_predicates = sorted(simple_predicates, key=str)

        for pred in simple_predicates:
            yield pred

        CP = predicates.CompoundPredicate

        for i in range(2, compound_length + 1):
            compound_predicates = itertools.combinations(simple_predicates, i)
            for compound_predicate in compound_predicates:
                yield CP(compound_predicate)

    def comparisons(self, predicates, simple_cover):
        compounder = self.Compounder(simple_cover)
        comparison_count = {}

        for pred in predicates:
            if len(pred) > 1:
                estimate = self.estimate(compounder(pred))
            else:
                estimate = self.estimate(simple_cover[pred])

            comparison_count[pred] = estimate

        return comparison_count

    class Compounder(object):
        def __init__(self, cover):
            self.cover = cover
            self._cached_predicate = None
            self._cached_cover = None

        def __call__(self, compound_predicate):
            a, b = compound_predicate[:-1], compound_predicate[-1]

            if len(a) > 1:
                if a == self._cached_predicate:
                    a_cover = self._cached_cover
                else:
                    a_cover = self._cached_cover = self(a)
                    self._cached_predicate = a
            else:
                a, = a
                a_cover = self.cover[a]

            return a_cover * self.cover[b]


class DedupeBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records, data):

        compound_length = 2

        N = sampled_records.original_length
        N_s = len(sampled_records)

        self.r = (N * (N - 1)) / (N_s * (N_s - 1))

        self.blocker = blocking.Blocker(predicates)
        self.blocker.indexAll(data)

        simple_cover = self.coveredPairs(self.blocker, sampled_records)
        compound_predicates = self.compound(simple_cover, compound_length)
        self.comparison_count = self.comparisons(compound_predicates,
                                                 simple_cover)

    @staticmethod
    def coveredPairs(blocker, records):
        cover = {}

        pair_enumerator = core.Enumerator()
        n_records = len(records)

        for predicate in blocker.predicates:
            pred_cover = collections.defaultdict(set)

            for id, record in viewitems(records):
                blocks = predicate(record)
                for block in blocks:
                    pred_cover[block].add(id)

            if not pred_cover:
                continue

            max_cover = max(len(v) for v in pred_cover.values())
            if max_cover == n_records:
                continue

            pairs = (pair_enumerator[pair]
                     for block in pred_cover.values()
                     for pair in itertools.combinations(sorted(block), 2))

            cover[predicate] = Counter(pairs)

        return cover

    def estimate(self, comparisons):
        # Result due to Stefano Allesina and Jacopo Grilli,
        # details forthcoming
        #
        # This estimates the total number of comparisons a blocking
        # rule will produce.
        #
        # While it is true that if we block together records 1 and 2 together
        # N times we have to pay the overhead of that blocking and
        # and there is some cost to each one of those N comparisons,
        # we are using a redundant-free scheme so we only make one
        # truly expensive computation for every record pair.
        #
        # So, how can we estimate how many expensive comparison a
        # predicate will lead to? In other words, how many unique record
        # pairs will be covered by a predicate?

        return self.r * comparisons.total


class RecordLinkBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records_1, sampled_records_2, data_2):

        compound_length = 2

        r_a = ((sampled_records_1.original_length) /
               len(sampled_records_1))
        r_b = ((sampled_records_2.original_length) /
               len(sampled_records_2))

        self.r = r_a * r_b

        self.blocker = blocking.Blocker(predicates)
        self.blocker.indexAll(data_2)

        simple_cover = self.coveredPairs(self.blocker,
                                         sampled_records_1,
                                         sampled_records_2)
        compound_predicates = self.compound(simple_cover, compound_length)

        self.comparison_count = self.comparisons(compound_predicates,
                                                 simple_cover)

    def coveredPairs(self, blocker, records_1, records_2):
        cover = {}

        pair_enumerator = core.Enumerator()

        for predicate in blocker.predicates:
            cover[predicate] = collections.defaultdict(lambda: (set(), set()))
            for id, record in viewitems(records_2):
                blocks = predicate(record, target=True)
                for block in blocks:
                    cover[predicate][block][1].add(id)

            current_blocks = set(cover[predicate])
            for id, record in viewitems(records_1):
                blocks = set(predicate(record))
                for block in blocks & current_blocks:
                    cover[predicate][block][0].add(id)

        for predicate, blocks in cover.items():
            pairs = {pair_enumerator[pair]
                     for A, B in blocks.values()
                     for pair in itertools.product(A, B)}
            cover[predicate] = Counter(pairs)

        return cover

    def estimate(self, comparisons):
        # For record pairs we only compare unique comparisons.
        #
        # I have no real idea of how to estimate the total number
        # of unique comparisons. Maybe the way to think about this
        # as the intersection of random multisets?
        #
        # In any case, here's the estimator we are using now.
        return self.r * comparisons.total


class BranchBound(object):
    def __init__(self, target, max_calls):
        self.calls = max_calls
        self.target = target
        self.cheapest_score = float('inf')
        self.original_cover = None

    def search(self, candidates, partial=()):
        if self.calls <= 0:
            return self.cheapest

        if self.original_cover is None:
            self.original_cover = candidates.copy()
            self.cheapest = candidates

        self.calls -= 1

        covered = self.covered(partial)
        score = self.score(partial)

        if covered >= self.target:
            if score < self.cheapest_score:
                self.cheapest = partial
                self.cheapest_score = score

        else:
            window = self.cheapest_score - score

            candidates = {p: cover
                          for p, cover in candidates.items()
                          if p.count < window}

            reachable = self.reachable(candidates) + covered

            if candidates and reachable >= self.target:

                order_by = functools.partial(self.order_by, candidates)

                best = max(candidates, key=order_by)

                remaining = self.uncovered_by(candidates,
                                              candidates[best])
                self.search(remaining, partial + (best,))
                del remaining

                reduced = self.remove_dominated(candidates, best)
                self.search(reduced, partial)
                del reduced

        return self.cheapest

    @staticmethod
    def order_by(candidates, p):
        return (len(candidates[p]), -p.count)

    @staticmethod
    def score(partial):
        return sum(p.count for p in partial)

    def covered(self, partial):
        if partial:
            return len(set.union(*(self.original_cover[p]
                                   for p in partial)))
        else:
            return 0

    @staticmethod
    def reachable(dupe_cover):
        if dupe_cover:
            return len(set.union(*dupe_cover.values()))
        else:
            return 0

    @staticmethod
    def remove_dominated(coverage, dominator):
        dominant_cover = coverage[dominator]

        for pred, cover in list(viewitems(coverage)):
            if (dominator.count <= pred.count and
                    dominant_cover >= cover):
                del coverage[pred]

        return coverage

    @staticmethod
    def uncovered_by(coverage, covered):
        remaining = {}
        for predicate, uncovered in viewitems(coverage):
            still_uncovered = uncovered - covered
            if still_uncovered:
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


class Counter(object):
    def __init__(self, iterable):
        if isinstance(iterable, Mapping):
            self._d = iterable
        else:
            d = collections.defaultdict(int)
            for elem in iterable:
                d[elem] += 1
            self._d = d

        self.total = sum(self._d.values())

    def __le__(self, other):
        return (self._d.keys() <= other._d.keys() and
                self.total <= other.total)

    def __eq__(self, other):
        return self._d == other._d

    def __len__(self):
        return len(self._d)

    def __mul__(self, other):

        if len(self) <= len(other):
            smaller, larger = self._d, other._d
        else:
            smaller, larger = other._d, self._d

        # it's meaningfully faster to check in the key dictview
        # of 'larger' than in the dict directly
        larger_keys = viewkeys(larger)

        common = {k: v * larger[k]
                  for k, v in viewitems(smaller)
                  if k in larger_keys}

        return Counter(common)


class Cover(object):
    def __init__(self, *args):
        if len(args) == 1:
            self._d, = args
        else:
            self._d = {}
            predicates, pairs = args
            self._cover(predicates, pairs)

    def __repr__(self):
        return 'Cover:' + str(self._d.keys())

    def _cover(self, predicates, pairs):
        for predicate in predicates:
            coverage = {i for i, (record_1, record_2)
                        in enumerate(pairs)
                        if (set(predicate(record_1)) &
                            set(predicate(record_2, target=True)))}
            if coverage:
                self._d[predicate] = coverage

    def compound(self, compound_length):
        simple_predicates = sorted(self._d, key=str)
        CP = predicates.CompoundPredicate

        for i in range(2, compound_length + 1):
            compound_predicates = itertools.combinations(simple_predicates, i)

            for compound_predicate in compound_predicates:
                a, b = compound_predicate[:-1], compound_predicate[-1]
                if len(a) == 1:
                    a = a[0]

                if a in self._d:
                    compound_cover = self._d[a] & self._d[b]
                    if compound_cover:
                        self._d[CP(compound_predicate)] = compound_cover

    def dominators(self, cost):
        def sort_key(x):
            return (-cost[x], len(self._d[x]))

        ordered_predicates = sorted(self._d, key=sort_key)
        dominants = {}

        for i, candidate in enumerate(ordered_predicates):
            candidate_match = self._d[candidate]
            candidate_cost = cost[candidate]

            for pred in ordered_predicates[(i + 1):]:
                other_match = self._d[pred]
                other_cost = cost[pred]
                better_or_equal = (other_match >= candidate_match and
                                   other_cost <= candidate_cost)
                if better_or_equal:
                    break
            else:
                dominants[candidate] = candidate_match

        self._d = dominants

    def __iter__(self):
        return iter(self._d)

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def __getitem__(self, k):
        return self._d[k]

    def copy(self):
        return Cover(self._d.copy())

    def update(self, *args, **kwargs):
        self._d.update(*args, **kwargs)

    def __eq__(self, other):
        return self._d == other._d

    def intersection_update(self, other):
        self._d = {k: self._d[k] for k in set(self._d) & set(other)}


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"  # noqa: E501
