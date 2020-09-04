#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

import itertools
import logging
import collections
import functools
from abc import ABC, abstractmethod

from . import blocking, core

logger = logging.getLogger(__name__)


class BlockLearner(ABC):
    def learn(self, matches, recall):
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        comparison_cover = self.comparison_cover  # type: ignore

        match_cover = Cover(self.blocker.predicates, matches)  # type: ignore
        match_cover.intersection_update(comparison_cover)

        coverable_dupes = set.union(*match_cover.values())
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]

        epsilon = int((1.0 - recall) * len(matches))

        if len(uncoverable_dupes) > epsilon:
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            epsilon = 0
        else:
            epsilon -= len(uncoverable_dupes)

        candidate_cover = self.generate_candidates(match_cover,
                                                   comparison_cover)

        searcher = BranchBound(len(coverable_dupes) - epsilon, 2500)
        final_predicates = searcher.search(candidate_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def generate_candidates(self,
                            match_cover: 'Cover',
                            comparison_cover: 'Cover') -> 'Cover':
        for pred in match_cover:
            pred.count = self.estimate(comparison_cover[pred])

        return match_cover

    @abstractmethod
    def estimate(self, comparisons):
        ...


class DedupeBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records, data):

        N = sampled_records.original_length
        N_s = len(sampled_records)

        self.r = (N * (N - 1)) / (N_s * (N_s - 1))

        self.blocker = blocking.Fingerprinter(predicates)
        self.blocker.index_all(data)

        self.comparison_cover = self.coveredPairs(self.blocker, sampled_records)

    @staticmethod
    def coveredPairs(blocker, records):
        cover = {}

        pair_enumerator = core.Enumerator()
        n_records = len(records)

        for predicate in blocker.predicates:
            pred_cover = collections.defaultdict(set)

            for id, record in records.items():
                blocks = predicate(record)
                for block in blocks:
                    pred_cover[block].add(id)

            if not pred_cover:
                continue

            max_cover = max(len(v) for v in pred_cover.values())
            if max_cover == n_records:
                continue

            pairs = {pair_enumerator[pair]
                     for block in pred_cover.values()
                     for pair in itertools.combinations(sorted(block), 2)}

            cover[predicate] = pairs

        return cover

    def estimate(self, comparisons):
        # Result due to Stefano Allesina and Jacopo Grilli,
        # details forthcoming
        #
        # This estimates the total number of comparisons a blocking
        # rule will produce.

        return self.r * len(comparisons)


class RecordLinkBlockLearner(BlockLearner):

    def __init__(self, predicates, sampled_records_1, sampled_records_2, data_2):

        r_a = ((sampled_records_1.original_length) /
               len(sampled_records_1))
        r_b = ((sampled_records_2.original_length) /
               len(sampled_records_2))

        self.r = r_a * r_b

        self.blocker = blocking.Fingerprinter(predicates)
        self.blocker.index_all(data_2)

        self.comparison_cover = self.coveredPairs(self.blocker,
                                                  sampled_records_1,
                                                  sampled_records_2)

    def coveredPairs(self, blocker, records_1, records_2):
        cover = {}

        pair_enumerator = core.Enumerator()

        for predicate in blocker.predicates:
            cover[predicate] = collections.defaultdict(lambda: (set(), set()))
            for id, record in records_2.items():
                blocks = predicate(record, target=True)
                for block in blocks:
                    cover[predicate][block][1].add(id)

            current_blocks = set(cover[predicate])
            for id, record in records_1.items():
                blocks = set(predicate(record))
                for block in blocks & current_blocks:
                    cover[predicate][block][0].add(id)

        for predicate, blocks in cover.items():
            pairs = {pair_enumerator[pair]
                     for A, B in blocks.values()
                     for pair in itertools.product(A, B)}
            cover[predicate] = pairs

        return cover

    def estimate(self, comparisons):
        # https://stats.stackexchange.com/a/465060/82

        return self.r * len(comparisons)


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

        for pred, cover in coverage.copy().items():
            if (dominator.count <= pred.count and
                    dominant_cover >= cover):
                del coverage[pred]

        return coverage

    @staticmethod
    def uncovered_by(coverage, covered):
        remaining = {}
        for predicate, uncovered in coverage.items():
            still_uncovered = uncovered - covered
            if still_uncovered:
                remaining[predicate] = still_uncovered

        return remaining


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
