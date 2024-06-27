from __future__ import annotations

import collections
import functools
import itertools
import logging
import math
import random
from abc import ABC
from typing import TYPE_CHECKING, overload
from warnings import warn

from . import blocking, branch_and_bound

if TYPE_CHECKING:
    from typing import Iterable, Literal, Sequence

    from ._typing import (
        ComparisonCover,
        ComparisonCoverInt,
        ComparisonCoverStr,
        Cover,
        Data,
        DataInt,
        DataStr,
    )
    from ._typing import RecordDictPairs as TrainingExamples
    from ._typing import RecordID, RecordIDPair
    from .predicates import Predicate


logger = logging.getLogger(__name__)


class BlockLearner(ABC):
    def learn(
        self,
        matches: TrainingExamples,
        recall: float,
        index_predicates: bool,
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
        match_cover = self.cover(matches, index_predicates=index_predicates)

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

        final_predicates = branch_and_bound.search(candidate_cover, target_cover, 2500)

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

    def cover(self, pairs: TrainingExamples, index_predicates: bool = True) -> Cover:
        predicate_cover = {}
        if index_predicates:
            predicates = self.blocker.predicates
        else:
            predicates = [
                pred for pred in self.blocker.predicates if not hasattr(pred, "index")
            ]

        for predicate in predicates:
            try:
                coverage = frozenset(
                    i
                    for i, (record_1, record_2) in enumerate(pairs)
                    if not predicate(record_1).isdisjoint(
                        predicate(record_2, target=True)
                    )
                )
            except AttributeError:
                warn(
                    f"the predicate {predicate.__name__} is not returning "
                    "a frozen set, this will soon be required behaviour",
                    DeprecationWarning,
                )
                coverage = frozenset(
                    i
                    for i, (record_1, record_2) in enumerate(pairs)
                    if not frozenset(predicate(record_1)).isdisjoint(
                        predicate(record_2, target=True)
                    )
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

    @overload
    @staticmethod
    def coveredPairs(
        blocker: blocking.Fingerprinter, records: DataInt
    ) -> ComparisonCoverInt: ...

    @overload
    @staticmethod
    def coveredPairs(
        blocker: blocking.Fingerprinter, records: DataStr
    ) -> ComparisonCoverStr: ...

    @staticmethod
    def coveredPairs(blocker: blocking.Fingerprinter, records):
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
    @overload
    def __init__(
        self,
        predicates: Iterable[Predicate],
        sampled_records_1: DataInt,
        sampled_records_2: DataInt,
        data_2: DataInt,
    ): ...

    @overload
    def __init__(
        self,
        predicates: Iterable[Predicate],
        sampled_records_1: DataStr,
        sampled_records_2: DataStr,
        data_2: DataStr,
    ): ...

    def __init__(
        self,
        predicates,
        sampled_records_1,
        sampled_records_2,
        data_2,
    ):
        self.blocker = blocking.Fingerprinter(predicates)
        self.blocker.index_all(data_2)

        self.comparison_cover = self.coveredPairs(
            self.blocker, sampled_records_1, sampled_records_2
        )

    @overload
    def coveredPairs(
        self, blocker: blocking.Fingerprinter, records_1: DataInt, records_2: DataInt
    ) -> ComparisonCoverInt: ...

    @overload
    def coveredPairs(
        self, blocker: blocking.Fingerprinter, records_1: DataStr, records_2: DataStr
    ) -> ComparisonCoverStr: ...

    def coveredPairs(self, blocker, records_1, records_2):
        cover: dict[Predicate, dict[str, tuple[set[RecordID], set[RecordID]]]] = {}
        pair_cover = {}
        n_records_1 = len(records_1)
        n_records_2 = len(records_2)

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

        cover = {
            predicate: blocks
            for predicate, blocks in cover.items()
            if not any(
                len(A) == n_records_1 and len(B) == n_records_2
                for A, B in blocks.values()
            )
        }

        for predicate, blocks in cover.items():
            pairs = frozenset(
                pair for A, B in blocks.values() for pair in itertools.product(A, B)
            )
            if pairs:
                pair_cover[predicate] = pairs

        return pair_cover


class InfiniteSet:
    def __and__(self, item):
        return item

    def __rand__(self, item):
        return item


class Resampler:
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

    @functools.lru_cache
    def __call__(self, iterable: Iterable[int]) -> frozenset[int]:
        result = itertools.chain.from_iterable(
            self.replacements[k] for k in iterable if k in self.replacements
        )
        return frozenset(result)
