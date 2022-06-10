from __future__ import annotations

import collections
import functools
import itertools
import logging
import math
import random
from abc import ABC
from typing import TYPE_CHECKING

from . import blocking

if TYPE_CHECKING:
    from typing import Any, Iterable, Mapping, Sequence

    from ._typing import ComparisonCover, Cover, Data, Literal
    from ._typing import RecordDictPairs as TrainingExamples
    from ._typing import RecordID, RecordIDPair
    from .predicates import Predicate


logger = logging.getLogger(__name__)


class BlockLearner(ABC):
    def learn(
        self,
        matches: TrainingExamples,
        recall: float,
        candidate_types: Literal["simple", "random forest"] = "simple",
    ) -> tuple[Predicate, ...]:
        """
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules.
        """
        assert (
            matches
        ), "You must supply at least one pair of matching records to learn blocking rules."

        comparison_cover = self.comparison_cover
        match_cover = self.cover(matches)

        for key in list(match_cover.keys() - comparison_cover.keys()):
            del match_cover[key]

        coverable_dupes = frozenset.union(*match_cover.values())
        uncoverable_dupes = [
            pair for i, pair in enumerate(matches) if i not in coverable_dupes
        ]

        target_cover = int(recall * len(matches))

        if len(coverable_dupes) < target_cover:
            logger.debug(uncoverable_dupes)
            target_cover = len(coverable_dupes)

        if candidate_types == "simple":
            candidate_cover = self.simple_candidates(match_cover, comparison_cover)
        elif candidate_types == "random forest":
            candidate_cover = self.random_forest_candidates(
                match_cover, comparison_cover
            )
        else:
            raise ValueError("candidate_type is not valid")

        searcher = BranchBound(target_cover, 2500)
        final_predicates = searcher.search(candidate_cover)

        logger.info("Final predicate set:")
        for predicate in final_predicates:
            logger.info(predicate)

        return final_predicates

    def simple_candidates(
        self, match_cover: Cover, comparison_cover: ComparisonCover
    ) -> Cover:
        candidates = {}
        for predicate, coverage in match_cover.items():
            predicate.cover_count = len(comparison_cover[predicate])
            candidates[predicate] = coverage.copy()

        return candidates

    def random_forest_candidates(
        self,
        match_cover: Cover,
        comparison_cover: ComparisonCover,
        K: int | None = None,
    ) -> Cover:
        predicates = list(match_cover)
        matches = list(frozenset.union(*match_cover.values()))
        pred_sample_size = max(int(math.sqrt(len(predicates))), 5)
        candidates = {}
        if K is None:
            K = max(math.floor(math.log10(len(matches))), 1)

        n_samples = 5000
        for _ in range(n_samples):
            sample_predicates = random.sample(predicates, pred_sample_size)
            resampler = Resampler(matches)
            sample_match_cover = {
                pred: resampler(pairs) for pred, pairs in match_cover.items()
            }

            # initialize variables that will be
            # the base for the constructing k-conjunctions
            candidate = None
            covered_comparisons: frozenset[RecordIDPair] | InfiniteSet = InfiniteSet()
            covered_matches: frozenset[int] | InfiniteSet = InfiniteSet()
            covered_sample_matches = InfiniteSet()

            def score(predicate: Predicate) -> float:
                try:
                    return len(
                        covered_sample_matches & sample_match_cover[predicate]
                    ) / len(covered_comparisons & comparison_cover[predicate])
                except ZeroDivisionError:
                    return 0.0

            for _ in range(K):
                next_predicate = max(sample_predicates, key=score)
                if candidate:
                    candidate += next_predicate
                else:
                    candidate = next_predicate

                covered_comparisons &= comparison_cover[next_predicate]
                candidate.cover_count = len(covered_comparisons)

                covered_matches &= match_cover[next_predicate]
                candidates[candidate] = covered_matches

                covered_sample_matches &= sample_match_cover[next_predicate]

                sample_predicates.remove(next_predicate)

        return candidates

    def cover(self, pairs: TrainingExamples) -> Cover:
        predicate_cover = {}
        for predicate in self.blocker.predicates:
            coverage = frozenset(
                i
                for i, (record_1, record_2) in enumerate(pairs)
                if (set(predicate(record_1)) & set(predicate(record_2, target=True)))
            )
            if coverage:
                predicate_cover[predicate] = coverage

        return predicate_cover

    blocker: blocking.Fingerprinter
    comparison_cover: ComparisonCover


class DedupeBlockLearner(BlockLearner):
    def __init__(
        self, predicates: Iterable[Predicate], sampled_records: Data, data: Data
    ):

        self.blocker = blocking.Fingerprinter(predicates)
        self.blocker.index_all(data)

        self.comparison_cover = self.coveredPairs(self.blocker, sampled_records)

    @staticmethod
    def coveredPairs(blocker: blocking.Fingerprinter, records: Data) -> ComparisonCover:
        cover = {}

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
                pair
                for block in pred_cover.values()
                for pair in itertools.combinations(sorted(block), 2)
            )
            if pairs:
                cover[predicate] = pairs

        return cover


class RecordLinkBlockLearner(BlockLearner):
    def __init__(
        self,
        predicates: Iterable[Predicate],
        sampled_records_1: Data,
        sampled_records_2: Data,
        data_2: Data,
    ):

        self.blocker = blocking.Fingerprinter(predicates)
        self.blocker.index_all(data_2)

        self.comparison_cover = self.coveredPairs(
            self.blocker, sampled_records_1, sampled_records_2
        )

    def coveredPairs(
        self, blocker: blocking.Fingerprinter, records_1: Data, records_2: Data
    ) -> ComparisonCover:
        cover: dict[Predicate, dict[str, tuple[set[RecordID], set[RecordID]]]] = {}
        pair_cover = {}

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
                pair for A, B in blocks.values() for pair in itertools.product(A, B)
            )
            if pairs:
                pair_cover[predicate] = pairs

        return pair_cover


class BranchBound(object):
    def __init__(self, target: int, max_calls: int) -> None:
        self.target: int = target
        self.calls: int = max_calls

        self.cheapest_score: float = float("inf")
        self.original_cover: Cover = {}
        self.cheapest: tuple[Predicate, ...] = ()

    def search(
        self, candidates: Cover, partial: tuple[Predicate, ...] = ()
    ) -> tuple[Predicate, ...]:
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

            candidates = {
                p: cover for p, cover in candidates.items() if p.cover_count < window
            }

            reachable = self.reachable(candidates) + covered

            if candidates and reachable >= self.target:

                order_by = functools.partial(self.order_by, candidates)

                best = max(candidates, key=order_by)

                remaining = self.uncovered_by(candidates, candidates[best])
                try:
                    self.search(remaining, partial + (best,))
                except RecursionError:
                    return self.cheapest

                del remaining

                reduced = self.remove_dominated(candidates, best)

                try:
                    self.search(reduced, partial)
                except RecursionError:
                    return self.cheapest

                del reduced

        return self.cheapest

    @staticmethod
    def order_by(
        candidates: Mapping[Predicate, Sequence[Any]], p: Predicate
    ) -> tuple[int, float]:
        return (len(candidates[p]), -p.cover_count)

    @staticmethod
    def score(partial: Iterable[Predicate]) -> float:
        return sum(p.cover_count for p in partial)

    def covered(self, partial: tuple[Predicate, ...]) -> int:
        if partial:
            return len(frozenset.union(*(self.original_cover[p] for p in partial)))
        else:
            return 0

    @staticmethod
    def reachable(dupe_cover: Mapping[Any, frozenset[int]]) -> int:
        if dupe_cover:
            return len(frozenset.union(*dupe_cover.values()))
        else:
            return 0

    @staticmethod
    def remove_dominated(coverage: Cover, dominator: Predicate) -> Cover:
        dominant_cover = coverage[dominator]

        for pred, cover in coverage.copy().items():
            if dominator.cover_count <= pred.cover_count and dominant_cover >= cover:
                del coverage[pred]

        return coverage

    @staticmethod
    def uncovered_by(
        coverage: Mapping[Any, frozenset[int]], covered: frozenset[int]
    ) -> dict[Any, frozenset[int]]:
        remaining = {}
        for predicate, uncovered in coverage.items():
            still_uncovered = uncovered - covered
            if still_uncovered:
                remaining[predicate] = still_uncovered

        return remaining


class InfiniteSet(object):
    def __and__(self, item):  # type: ignore[no-untyped-def]
        return item

    def __rand__(self, item):  # type: ignore[no-untyped-def]
        return item


class Resampler(object):
    def __init__(self, sequence: Sequence[int]):

        sampled = random.choices(sequence, k=len(sequence))

        c = collections.Counter(sampled)
        max_value = len(sequence) + 1

        self.replacements: dict[int, list[int]] = {}
        for k, v in c.items():
            self.replacements[k] = [v]
            if v > 1:
                for _ in range(v - 1):
                    self.replacements[k].append(max_value)
                    max_value += 1

    @functools.lru_cache()
    def __call__(self, iterable: Iterable[int]) -> frozenset[int]:

        result = itertools.chain.from_iterable(
            self.replacements[k] for k in iterable if k in self.replacements
        )
        return frozenset(result)
