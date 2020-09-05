#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

import itertools
import logging
import collections
import functools
import random
from abc import ABC, abstractmethod
import math

from . import blocking, core
from .predicates import CompoundPredicate

logger = logging.getLogger(__name__)


class BlockLearner(ABC):
    def learn(self, matches, recall):
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        comparison_cover = self.comparison_cover  # type: ignore
        match_cover = self.cover(matches)  # type: ignore

        for key in list(match_cover.keys() - comparison_cover.keys()):
            del match_cover[key]

        coverable_dupes = frozenset.union(*match_cover.values())
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]

        epsilon = int((1.0 - recall) * len(matches))

        if len(uncoverable_dupes) > epsilon:
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            epsilon = 0
        else:
            epsilon -= len(uncoverable_dupes)

        candidate_cover = self.generate_candidates_rf(match_cover,
                                                      comparison_cover)

        searcher = BranchBound(len(coverable_dupes) - epsilon, 2500)
        final_predicates = searcher.search(candidate_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def generate_candidates_bilenko(self,
                                    match_cover: dict,
                                    comparison_cover: dict) -> dict:
        predicates = list(match_cover)
        candidates = {}
        K = 3

        for i, predicate in enumerate(predicates):
            current_match_cover = match_cover[predicate]
            current_comparison_cover = comparison_cover[predicate]
            predicate.count = self.estimate(current_comparison_cover)
            candidates[predicate] = current_match_cover
            remaining = predicates
            predicate = CompoundPredicate(predicate,)
            for _ in range(K):
                if not remaining:
                    break
                best_p = max(remaining,
                             key=lambda x: (len(current_match_cover & match_cover[x]) /
                                            (self.estimate(current_comparison_cover & comparison_cover[x]) or float('inf'))))
                predicate = CompoundPredicate(predicate + (best_p,))
                current_match_cover &= match_cover[best_p]
                current_comparison_cover &= comparison_cover[best_p]
                predicate.count = self.estimate(current_comparison_cover)
                candidates[predicate] = current_match_cover
                remaining.remove(best_p)

        return candidates

    def generate_candidates_rf(self,
                               match_cover: dict,
                               comparison_cover: dict) -> dict:
        predicates = list(match_cover)
        matches = list(frozenset.union(*match_cover.values()))
        n_pred_sample_size = max(int(math.sqrt(len(predicates))), 5)
        candidates = {}
        K = 3

        n_candidates = 10000
        for _ in range(n_candidates):
            sample_predicates = random.sample(predicates,
                                              n_pred_sample_size)
            expander = Expander(random.choices(matches, k=len(matches)), max(matches))
            current_match_cover = InfiniteSet()
            real_match_cover = InfiniteSet()
            current_comparison_cover = InfiniteSet()
            predicate = CompoundPredicate()
            for _ in range(K):
                best_p = max(sample_predicates,
                             key=lambda x: (len((current_match_cover & expander(match_cover[x]))) /
                                            (self.estimate(current_comparison_cover & comparison_cover[x]) or float('inf'))))
                predicate = CompoundPredicate(predicate + (best_p,))
                current_match_cover &= expander(match_cover[best_p])
                real_match_cover &= match_cover[best_p]
                current_comparison_cover &= comparison_cover[best_p]
                predicate.count = self.estimate(current_comparison_cover)
                candidates[predicate] = real_match_cover
                sample_predicates.remove(best_p)

        return candidates

    def cover(self, pairs):
        predicate_cover = {}
        for predicate in self.blocker.predicates:  # type: ignore
            coverage = frozenset(
                i for i, (record_1, record_2)
                in enumerate(pairs)
                if (set(predicate(record_1)) &
                    set(predicate(record_2, target=True))))
            if coverage:
                predicate_cover[predicate] = coverage

        return predicate_cover

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

            pairs = frozenset(
                pair_enumerator[pair]
                for block in pred_cover.values()
                for pair in itertools.combinations(sorted(block), 2))
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
            pairs = frozenset(
                pair_enumerator[pair]
                for A, B in blocks.values()
                for pair in itertools.product(A, B))
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
            self.cheapest = tuple(candidates)

        self.calls -= 1

        covered = self.covered(partial)
        score = self.score(partial)

        if covered >= self.target:
            if score < self.cheapest_score:
                self.cheapest = partial
                self.cheapest_score = score
                print(score)

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
            return len(frozenset.union(*(self.original_cover[p]
                                         for p in partial)))
        else:
            return 0

    @staticmethod
    def reachable(dupe_cover):
        if dupe_cover:
            return len(frozenset.union(*dupe_cover.values()))
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


class InfiniteSet(object):

    def __and__(self, item):
        return item

    def __rand__(self, item):
        return item


class Expander(object):

    def __init__(self, iterable, max_value):
        c = collections.Counter(iterable)
        max_value += 1

        self.replacements = {}
        for k, v in c.items():
            self.replacements[k] = [v]
            if v > 1:
                for _ in range(v - 1):
                    self.replacements[k].append(max_value)
                    max_value += 1

    def __call__(self, iterable):

        result = itertools.chain.from_iterable(self.replacements[k]
                                               for k in iterable
                                               if k in self.replacements)
        return frozenset(result)


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"  # noqa: E501
