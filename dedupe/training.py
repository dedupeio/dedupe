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

from typing import (Dict, Sequence, Iterable, Tuple, List,
                    Union, FrozenSet)

from . import blocking, core
from .predicates import Predicate

logger = logging.getLogger(__name__)

Cover = Dict[Predicate, FrozenSet[int]]


class BlockLearner(ABC):
    def learn(self, matches, recall, candidate_types='simple'):
        '''
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        '''
        comparison_cover = self.comparison_cover
        match_cover = self.cover(matches)

        for key in list(match_cover.keys() - comparison_cover.keys()):
            del match_cover[key]

        coverable_dupes = frozenset.union(*match_cover.values())
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]

        target_cover = int(recall * len(matches))

        if len(coverable_dupes) < target_cover:
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            target_cover = len(coverable_dupes)

        if candidate_types == 'simple':
            candidate_cover = self.simple_candidates(match_cover,
                                                     comparison_cover)
        elif candidate_types == 'random forest':
            candidate_cover = self.random_forest_candidates(match_cover,
                                                            comparison_cover)
        else:
            raise ValueError('candidate_type is not valid')

        searcher = BranchBound(target_cover, 2500)
        final_predicates = searcher.search(candidate_cover)

        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def simple_candidates(self,
                          match_cover: Cover,
                          comparison_cover: Cover) -> Cover:
        candidates = {}
        for predicate, coverage in match_cover.items():
            predicate.count = self.estimate(comparison_cover[predicate])  # type: ignore
            candidates[predicate] = coverage.copy()

        return candidates

    def random_forest_candidates(self,
                                 match_cover: Cover,
                                 comparison_cover: Cover) -> Cover:
        predicates = list(match_cover)
        matches = list(frozenset.union(*match_cover.values()))
        pred_sample_size = max(int(math.sqrt(len(predicates))), 5)
        candidates = {}
        K = 3

        n_samples = 5000
        for _ in range(n_samples):
            sample_predicates = random.sample(predicates,
                                              pred_sample_size)
            resampler = Resampler(matches)
            sample_match_cover = {pred: resampler(pairs)
                                  for pred, pairs
                                  in match_cover.items()}

            # initialize variables that will be
            # the base for the constructing k-conjunctions
            candidate = None
            covered_comparisons = InfiniteSet()
            covered_matches: Union[FrozenSet[int], InfiniteSet] = InfiniteSet()
            covered_sample_matches = InfiniteSet()

            def score(predicate: Predicate) -> float:
                try:
                    return (len(covered_sample_matches &
                                sample_match_cover[predicate]) /
                            self.estimate(covered_comparisons &
                                          comparison_cover[predicate]))
                except ZeroDivisionError:
                    return 0.

            for _ in range(K):
                next_predicate = max(sample_predicates, key=score)
                if candidate:
                    candidate += next_predicate
                else:
                    candidate = next_predicate

                covered_comparisons &= comparison_cover[next_predicate]
                candidate.count = self.estimate(covered_comparisons)  # type: ignore

                covered_matches &= match_cover[next_predicate]
                candidates[candidate] = covered_matches

                covered_sample_matches &= sample_match_cover[next_predicate]

                sample_predicates.remove(next_predicate)

        return candidates

    def cover(self, pairs) -> Cover:
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
    def estimate(self, comparisons) -> float:
        ...

    blocker: blocking.Fingerprinter
    comparison_cover: Cover


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
    def __init__(self, target: int, max_calls: int) -> None:
        self.target: int = target
        self.calls: int = max_calls

        self.cheapest_score: float = float('inf')
        self.original_cover: Cover = {}
        self.cheapest: Tuple[Predicate, ...] = ()

    def search(self,
               candidates: Cover,
               partial: Tuple[Predicate, ...] = ()) -> Tuple[Predicate, ...]:
        if self.calls <= 0:
            return self.cheapest

        if not self.original_cover:
            self.original_cover = candidates.copy()

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
                          if p.count < window}  # type: ignore

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
    def order_by(candidates: Cover, p: Predicate) -> Tuple[int, float]:
        return (len(candidates[p]), -p.count)  # type: ignore

    @staticmethod
    def score(partial: Iterable[Predicate]) -> float:
        return sum(p.count for p in partial)  # type: ignore

    def covered(self, partial: Tuple[Predicate, ...]) -> int:
        if partial:
            return len(frozenset.union(*(self.original_cover[p]
                                         for p in partial)))
        else:
            return 0

    @staticmethod
    def reachable(dupe_cover: Cover) -> int:
        if dupe_cover:
            return len(frozenset.union(*dupe_cover.values()))
        else:
            return 0

    @staticmethod
    def remove_dominated(coverage: Cover, dominator: Predicate) -> Cover:
        dominant_cover = coverage[dominator]

        for pred, cover in coverage.copy().items():
            if (dominator.count <= pred.count and  # type: ignore
                    dominant_cover >= cover):
                del coverage[pred]

        return coverage

    @staticmethod
    def uncovered_by(coverage: Cover, covered: frozenset) -> Cover:
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


class Resampler(object):

    def __init__(self, sequence: Sequence):

        sampled = random.choices(sequence, k=len(sequence))

        c = collections.Counter(sampled)
        max_value = len(sequence) + 1

        self.replacements: Dict[int, List[int]] = {}
        for k, v in c.items():
            self.replacements[k] = [v]
            if v > 1:
                for _ in range(v - 1):
                    self.replacements[k].append(max_value)
                    max_value += 1

    @functools.lru_cache()
    def __call__(self, iterable: Iterable) -> frozenset:

        result = itertools.chain.from_iterable(self.replacements[k]
                                               for k in iterable
                                               if k in self.replacements)
        return frozenset(result)


OUT_OF_PREDICATES_WARNING = "Ran out of predicates: Dedupe tries to find blocking rules that will work well with your data. Sometimes it can't find great ones, and you'll get this warning. It means that there are some pairs of true records that dedupe may never compare. If you are getting bad results, try increasing the `max_comparison` argument to the train method"  # noqa: E501
