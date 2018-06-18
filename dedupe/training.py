#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data
from __future__ import division
from future.utils import viewitems, viewvalues, viewkeys

import itertools
import logging
import collections

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
        self.pair_id = core.Enumerator()

        blocker = blocking.Blocker(predicates)
        blocker.indexAll(data)

        result = self.coveredPairs(blocker, sampled_records)
        self.total_cover, self.record_cover = result

        self.r = (sampled_records.original_length + 1) / len(sampled_records)

        self.blocker = blocking.Blocker(predicates)

        self._cached_estimates = {}

    @staticmethod
    def unroll(matches):  # pragma: no cover
        return unique((record for pair in matches for record in pair))

    def coveredPairs(self, blocker, records):
        cover = {}
        record_cover = {}

        for predicate in blocker.predicates:
            pred_cover = collections.defaultdict(set)
            covered_records = collections.defaultdict(int)

            for id, record in viewitems(records):
                blocks = predicate(record)
                for block in blocks:
                    pred_cover[block].add(id)
                    covered_records[id] += 1

            pairs = (pair
                     for block in pred_cover.values()
                     for pair in itertools.combinations(sorted(block), 2))

            cover[predicate] = Counter(pairs)
            record_cover[predicate] = Counter(covered_records)

        return cover, record_cover

    def estimate(self, comparisons, records):
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
        # Now, unfortunately, it's difficult to efficiently calculate
        # x_i * x_i for compound blocks.
        #
        # Perhaps suprisingly, it's much easier to calculate the
        # number of comparisons that a predicate generates on a
        # sample. This is
        #
        # D = sum(x_i * (x_i - 1)/2 for x_i in X)
        #   = 0.5 (sum(x_i * x_i for x_i in X) - sum(x_i for x_i in X))
        #
        # It turns out that
        #
        # C = r * r * D  + 0.5 * (r * r - r) * sum(x_i for x_i in X))

        r = self.r

        all_comparisons = (r * r * comparisons.total +
                           0.5 * (r * r - r) * records.total)

        # This estimator has a couple of problems.
        #
        # First, it asssumes that we have observered every block_key
        # in the sample that we will observe in the full data. This
        # does not seem like a terrible problem because block_keys
        # that do not appear in the sample will be fairly rare, so are
        # likely not to contribute very much to the total number of
        # comparisons. However, by neglecting this our estimate is
        # clearly biased to be less than the true value. I'm not sure
        # how to fix this right now, but it seems like would be a
        # reasonably small extension
        #
        # However, this estimate is for every single comparisons. While
        # it is true that if we block together records 1 and 2 together
        # N times we have to pay the overhead of that blocking and
        # and there is some cost to each one of those N comparisons,
        # we are using a redundant-free scheme so we only make one
        # truly expensive computation for every record pair.
        #
        # So, how can we estimate how many expensive comparison a
        # predicate will lead to? In other words, how many unique record
        # pairs will be covered by a predicate?
        #
        # I really don't know, right now. So we'll stick with a number
        # we can defend.

        return all_comparisons

    def comparisons(self, match_cover, compound_length):
        cover_compounder = Compounder(self.total_cover)
        record_compounder = Compounder(self.record_cover)
        comparison_count = {}

        for pred in sorted(match_cover, key=str):
            if pred in self._cached_estimates:
                estimated_comparisons = self._cached_estimates[pred]
            elif len(pred) > 1:
                estimated_comparisons = self.estimate(cover_compounder(pred),
                                                      record_compounder(pred))
            else:
                estimated_comparisons = self.estimate(self.total_cover[pred],
                                                      self.record_cover[pred])

            comparison_count[pred] = estimated_comparisons
            self._cached_estimates[pred] = estimated_comparisons

        return comparison_count


class RecordLinkBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records_1, sampled_records_2, data_2):
        self.pair_id = core.Enumerator()

        blocker = blocking.Blocker(predicates)
        blocker.indexAll(data_2)

        self.total_cover = self.coveredPairs(blocker,
                                             sampled_records_1,
                                             sampled_records_2)

        self.r_a = ((sampled_records_1.original_length) /
                    len(sampled_records_1))
        self.r_b = ((sampled_records_2.original_length) /
                    len(sampled_records_2))

        self.blocker = blocking.Blocker(predicates)

        self._cached_estimates = {}

    @staticmethod
    def unroll(matches):  # pragma: no cover
        return unique((record_2 for _, record_2 in matches))

    def coveredPairs(self, blocker, records_1, records_2):
        cover = {}

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
            pairs = {pair
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
        return len(comparisons) * self.r_a * self.r_b

    def comparisons(self, match_cover, compound_length):
        compounder = Compounder(self.total_cover)
        comparison_count = {}

        for pred in sorted(match_cover, key=str):
            if pred in self._cached_estimates:
                estimated_cached_estimates = self._cached_estimates[pred]
            elif len(pred) > 1:
                estimated_cached_estimates = self.estimate(compounder(pred))
            else:
                estimated_cached_estimates = self.estimate(self.total_cover[pred])

            comparison_count[pred] = estimated_cached_estimates
            self._cached_estimates[pred] = estimated_cached_estimates

        return comparison_count


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


class Counter(object):
    def __init__(self, iterable):
        if isinstance(iterable, Mapping):
            self._d = iterable
        else:
            d = collections.defaultdict(int)
            for elem in iterable:
                d[elem] += 1
            self._d = d

        self.total = sum(self.values())

    def __le__(self, other):
        return (self._d.keys() <= other._d.keys() and
                self.total <= other.total)

    def __eq__(self, other):
        return self._d == other._d

    def __len__(self):
        return len(self._d)

    def values(self):
        return viewvalues(self._d)

    def __mul__(self, other):

        if len(self) <= len(other):
            smaller, larger = self._d, other._d
        else:
            smaller, larger = other._d, self._d

        # it's meaningfully faster to check in the key dictview
        # of 'larger' than in the dict directly
        larger_keys = viewkeys(larger)

        common = {k: v * larger[k] for k, v in viewitems(smaller)
                  if k in larger_keys}

        return Counter(common)


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"  # noqa: E501
