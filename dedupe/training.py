#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data
from __future__ import division
from future.utils import viewitems, viewvalues

import itertools
import logging
import collections
import functools

from . import blocking, predicates, core

logger = logging.getLogger(__name__)


class BlockLearner(object):
    def learn(self, matches, recall):
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        compound_length = 2

        dupe_cover = cover(self.blocker, matches,
                           self.total_cover, compound_length)
        comparison_count = self.comparisons(dupe_cover, compound_length)

        dupe_cover = dominators(dupe_cover, comparison_count, comparison=True)

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

    def comparisons(self, match_cover, compound_length):
        compounder = Compounder(self.total_cover, self.record_cover)
        comparison_count = {}

        for pred in sorted(match_cover, key=str):
            if len(pred) > 1:
                comparison_count[pred] = self.estimate(*compounder(pred))
            else:
                comparison_count[pred] = self.estimate(self.total_cover[pred],
                                                       self.record_cover[pred])

        return comparison_count


class Compounder(object):
    def __init__(self, cover, record_cover):
        self.cover = cover
        self.record_cover = record_cover
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

        return a_cover & self.cover[b], self.record_cover[a] & self.record_cover[b]


class DedupeBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records, data):
        self.pair_id = core.Enumerator()

        blocker = blocking.Blocker(predicates)
        blocker.indexAll(data)

        self.total_cover, self.record_cover = self.coveredPairs(blocker, sampled_records)
        self.multiplier = sampled_records.original_length / len(sampled_records)

        self.original_length = sampled_records.original_length
        self.sample_size = len(sampled_records)

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches):  # pragma: no cover
        return unique((record for pair in matches for record in pair))

    def coveredPairs(self, blocker, records):
        cover = {}
        record_cover = {}

        for predicate in blocker.predicates:
            pred_cover = {}
            covered_records = collections.defaultdict(int)

            for id, record in viewitems(records):
                blocks = predicate(record)
                for block in blocks:
                    pred_cover.setdefault(block, set()).add(id)
                    covered_records[id] += 1

            pairs = (pair
                     for block in pred_cover.values()
                     for pair in itertools.combinations(sorted(block), 2))

            cover[predicate] = Counter(pairs)
            record_cover[predicate] = Counter(covered_records)

        return cover, record_cover

    def estimate(self, blocks, records):
        # We want to estimate the number of comparisons that a
        # predicate will generate on the full data.
        #
        # To start, we can estimate the number of records, k, that are
        # covered by a particular block key in the full data.
        #
        # Let x the number of records in a sample that have the same
        # block key. Because the sample is drawn witout replacement, x
        # is a random variable drawn from a hypergeometric distribution.
        #
        # Let M be the size of full data, and n be the size of the sample
        #
        # The Maximum likelihood estimator of k is
        #
        # k = math.floor((M + 1)* x/n)
        #
        # we will relax this a bit and have
        # k = (M + 1)/n * x
        #
        # We can then estimate the number of comparisons, c, that this block
        # key will generate on the full data.
        #
        # c = k(k-1)/2
        # c = 0.5 * ((M + 1)/n * x) * ((M + 1)/n * (x - 1))
        #
        # let r = (M + 1)/n
        #
        # c = 0.5 * r * x * (r * (x -1))
        #   = 0.5 * (r * r * x * x  - r * x)
        #
        # Every blocking predicate will generate multiple block
        # keys. Let x_i be the number of records covered by the ith
        # block key produced by a blocking predicate, and c_i be the
        # associated number of estimated comparisons
        #
        # We estimate the total number of record comparisons, C, generated
        # by a blocking predicate as
        #
        # C = sum(c_i)
        #   = 0.5 * (r * r * sum(x_i * x_i for x_i in X) -
        #            r * sum(x_i for x_i in X))
        #
        # Now, unfortunately, it's difficult to efficiently x_i for
        # compound blocks.
        #
        # Perhaps suprisingly, it's much easier to
        # calculate the number of comparisons that a predicate
        # generates on a sample. This is
        #
        # D = sum(x_i * (x_i - 1)/2 for x_i in X)
        #   = 0.5 (sum(x_i * x_i for x_i in X) - sum(x_i for x_i in X))
        #
        # It turns out that
        #
        # C = r * r * 2 * D  + 0.5 * (r * r - r) * sum(x_i for x_i in X))

        r = (self.original_length + 1)/self.sample_size

        abundance = (r * r * sum(blocks.values()) +
                     0.5 * (r * r - r) * sum(records.values()))

        return abundance


class RecordLinkBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records_1, sampled_records_2, data_2):
        self.pair_id = core.Enumerator()

        blocker = blocking.Blocker(predicates)
        blocker.indexAll(data_2)

        self.total_cover = self.coveredPairs(blocker,
                                             sampled_records_1,
                                             sampled_records_2)

        self.multiplier_1 = (sampled_records_1.original_length /
                             len(sampled_records_1))
        self.multiplier_2 = (sampled_records_2.original_length /
                             len(sampled_records_2))

        self.blocker = blocking.Blocker(predicates)

    @staticmethod
    def unroll(matches):  # pragma: no cover
        return unique((record_2 for _, record_2 in matches))

    def coveredPairs(self, blocker, records_1, records_2):
        cover = {}

        for predicate in blocker.predicates:
            cover[predicate] = {}
            for id, record in viewitems(records_2):
                blocks = predicate(record, target=True)
                for block in blocks:
                    cover[predicate].setdefault(
                        block, (set(), set()))[1].add(id)

            current_blocks = set(cover[predicate])
            for id, record in viewitems(records_1):
                blocks = set(predicate(record))
                for block in blocks & current_blocks:
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

                def score(p):
                    return (len(candidates[p]), -p.count)

                best = max(candidates, key=score)

                remaining = remaining_cover(candidates,
                                            candidates[best])
                self.search(remaining, partial + (best,))
                del remaining

                reduced = self.dominates(candidates, best)
                self.search(reduced, partial)
                del reduced

        return self.cheapest

    def score(self, partial):
        return sum(p.count for p in partial)

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
        dominant_cover = coverage[dominator]

        for pred, cover in list(viewitems(coverage)):
            if (dominator.count <= pred.count and
                    dominant_cover >= cover):
                del coverage[pred]

        return coverage


def cover(blocker, pairs, total_cover, compound_length):  # pragma: no cover
    cover = coveredPairs(blocker.predicates, pairs)
    cover = dominators(cover, total_cover)
    cover = compound(cover, compound_length)
    cover = remaining_cover(cover)

    return cover


def coveredPairs(predicates, pairs):
    cover = {}

    for predicate in predicates:
        coverage = {i for i, (record_1, record_2)
                    in enumerate(pairs)
                    if (set(predicate(record_1)) &
                        set(predicate(record_2, target=True)))}
        if coverage:
            cover[predicate] = coverage

    return cover


def compound(cover, compound_length):
    simple_predicates = sorted(cover, key=str)
    CP = predicates.CompoundPredicate

    for i in range(2, compound_length + 1):
        compound_predicates = itertools.combinations(simple_predicates, i)

        for compound_predicate in compound_predicates:
            a, b = compound_predicate[:-1], compound_predicate[-1]
            if len(a) == 1:
                a = a[0]

            if a in cover:
                compound_cover = cover[a] & cover[b]
                if compound_cover:
                    cover[CP(compound_predicate)] = compound_cover

    return cover


def remaining_cover(coverage, covered=None):
    if covered is None:
        covered = set()
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


def dominators(match_cover, total_cover, comparison=False):
    if comparison:
        def sort_key(x):
            return (-total_cover[x], len(match_cover[x]))
    else:
        def sort_key(x):
            return (len(match_cover[x]), -len(total_cover[x]))

    ordered_predicates = sorted(match_cover, key=sort_key)
    dominants = {}

    for i, candidate in enumerate(ordered_predicates, 1):
        match = match_cover[candidate]
        total = total_cover[candidate]

        if not any((match_cover[pred] >= match and
                    total_cover[pred] <= total)
                   for pred in ordered_predicates[i:]):
            dominants[candidate] = match

    return dominants


@functools.total_ordering
class Counter(collections.Counter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get('cache', True):
            self.ones = set()
            self.multiples = {}
            for k, v in self.items():
                if v == 1:
                    self.ones.add(k)
                else:
                    self.multiples[k] = v
            self._key_set = set(self.multiples.keys())

    def __le__(self, other):
        return self.keys() <= other.keys()

    def __lt__(self, other):
        return self.keys() < other.keys()

    def __eq__(self, other):
        return self == other

    def __and__(self, other):

        common = dict.fromkeys(self.ones, 1)

        both_multiple = ((k, self.multiples[k] * other.multiples[k])
                         for k in self._key_set & other._key_set)
        other_ones = ((k, self[k]) for k in self._key_set & other.ones)
        self_ones = ((k, other[k]) for k in self.ones & other._key_set)

        common.update(itertools.chain(both_multiple,
                                      other_ones,
                                      self_ones))

        return common

        


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"  # noqa: E501
