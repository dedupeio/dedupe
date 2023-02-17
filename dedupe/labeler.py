from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload
from warnings import warn

import numpy
import numpy.typing
import sklearn.linear_model

import dedupe.core as core
import dedupe.training as training

if TYPE_CHECKING:
    from typing import Dict, Iterable, Literal, Mapping

    from dedupe._typing import (
        Data,
        DataInt,
        DataStr,
        FeaturizerFunction,
        Labels,
        LabelsLike,
    )
    from dedupe._typing import RecordDictPair as TrainingExample
    from dedupe._typing import RecordDictPairs as TrainingExamples
    from dedupe._typing import RecordIDPair
    from dedupe.predicates import Predicate


logger = logging.getLogger(__name__)


class HasCandidates:
    """Has a list of pairs that we could ask the user to label."""

    _candidates: TrainingExamples

    @property
    def candidates(self) -> TrainingExamples:
        return self._candidates

    def __len__(self) -> int:
        return len(self.candidates)


class Learner(ABC, HasCandidates):
    """A single learner that is used by DisagreementLearner."""

    _fitted: bool = False

    @abstractmethod
    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        """Train on the given data."""

    @abstractmethod
    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        """For each of self.candidates, return our current guess [0,1] of if a match."""

    @abstractmethod
    def remove(self, index: int) -> None:
        """Remove a pair from self.candidates."""

    @staticmethod
    def _verify_fit_args(pairs: TrainingExamples, y: LabelsLike) -> list[Literal[0, 1]]:
        """Helper method to verify the arguments given to fit()"""
        if len(pairs) == 0:
            raise ValueError("pairs must have length of at least 1")
        y = list(y)
        if len(pairs) != len(y):
            raise ValueError(
                f"pairs and y must be same length. Got {len(pairs)} and {len(y)}"
            )
        return y


class MatchLearner(Learner):
    def __init__(self, featurizer: FeaturizerFunction, candidates: TrainingExamples):
        self._featurizer = featurizer
        self._candidates = candidates.copy()
        self._classifier = sklearn.linear_model.LogisticRegression()
        self._features = self._featurizer(self.candidates)

    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        y = self._verify_fit_args(pairs, y)
        self._classifier.fit(self._featurizer(pairs), numpy.array(y))
        self._fitted = True

    def remove(self, index: int) -> None:
        self._candidates.pop(index)
        self._features = numpy.delete(self._features, index, axis=0)

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        if not self._fitted:
            raise ValueError("Must call fit() before candidate_scores()")
        return self._classifier.predict_proba(self._features)[:, 1].reshape(-1, 1)


class BlockLearner(Learner):
    block_learner: training.BlockLearner

    def __init__(self):
        self.current_predicates: tuple[Predicate, ...] = ()
        self._cached_scores: numpy.typing.NDArray[numpy.float_] | None = None
        self._old_dupes: TrainingExamples = []

    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        y = self._verify_fit_args(pairs, y)
        dupes = [pair for label, pair in zip(y, pairs) if label]

        new_dupes = [pair for pair in dupes if pair not in self._old_dupes]
        new_uncovered = not all(self._predict(new_dupes))

        if new_uncovered:
            self.current_predicates = self.block_learner.learn(
                dupes, recall=1.0, index_predicates=True, candidate_types="simple"
            )
            self._cached_scores = None
            self._old_dupes = dupes
        self._fitted = True

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        if not self._fitted:
            raise ValueError("Must call fit() before candidate_scores()")
        if self._cached_scores is None:
            labels = self._predict(self.candidates)
            self._cached_scores = numpy.array(labels).reshape(-1, 1)

        return self._cached_scores

    def learn_predicates(
        self, dupes: TrainingExamples, recall: float, index_predicates: bool
    ) -> tuple[Predicate, ...]:
        return self.block_learner.learn(
            dupes,
            recall=recall,
            index_predicates=index_predicates,
            candidate_types="random forest",
        )

    def _predict(self, pairs: TrainingExamples) -> Labels:
        labels: Labels = []
        for record_1, record_2 in pairs:
            for predicate in self.current_predicates:
                keys = predicate(record_2, target=True)
                try:
                    overlap = not keys.isdisjoint(predicate(record_1))
                except AttributeError:
                    warn(
                        f"the predicate {predicate.__name__} is not returning "
                        "a frozen set, this will soon be required behaviour",
                        DeprecationWarning,
                    )
                    overlap = not frozenset(keys).isdisjoint(predicate(record_1))
                if overlap:
                    labels.append(1)
                    break
            else:
                labels.append(0)

        return labels

    def remove(self, index: int) -> None:
        self._candidates.pop(index)
        if self._cached_scores is not None:
            self._cached_scores = numpy.delete(self._cached_scores, index, axis=0)

    def _sample_indices(
        self, sample_size: int, max_cover: int
    ) -> Iterable[RecordIDPair]:
        weights: Dict[RecordIDPair, float] = {}
        for predicate, covered in self.block_learner.comparison_cover.items():
            # each predicate gets to vote for every record pair it covers. the
            # strength of that vote is in inverse proportion to the number of
            # records the predicate covers.
            #
            # if a predicate only covers a few record pairs, the value of
            # the vote it puts on those few pairs will be worth more than
            # a predicate that covers almost all the record pairs
            if len(covered) < max_cover:
                weight = 1 / len(covered)
                for pair in covered:
                    weights[pair] = weights.get(pair, 0.0) + weight

        sample_ids: Iterable[RecordIDPair]
        if sample_size < len(weights):
            # consider using a reservoir sampling strategy, which would
            # be more memory efficient and probably about as fast
            normalized_weights = numpy.fromiter(
                weights.values(), dtype=float, count=len(weights)
            ) / sum(weights.values())
            rng = numpy.random.default_rng()
            sample_indices = rng.choice(
                len(weights), size=sample_size, replace=False, p=normalized_weights
            )
            keys = list(weights.keys())
            sample_ids = ((keys[i][0], keys[i][1]) for i in sample_indices)
        else:
            sample_ids = weights.keys()

        return sample_ids


def _filter_canopy_predicates(
    predicates: Iterable[Predicate], canopies: bool
) -> set[Predicate]:
    result = set()
    for predicate in predicates:
        if hasattr(predicate, "index"):
            is_canopy = hasattr(predicate, "canopy")
            if is_canopy == canopies:
                result.add(predicate)
        else:
            result.add(predicate)
    return result


class DedupeBlockLearner(BlockLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        data: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()

        N_SAMPLED_RECORDS = 5000
        N_SAMPLED_RECORD_PAIRS = 10000

        index_data = sample_records(data, 50000)
        sampled_records = sample_records(index_data, N_SAMPLED_RECORDS)

        preds = _filter_canopy_predicates(candidate_predicates, canopies=True)
        self.block_learner = training.DedupeBlockLearner(
            preds, sampled_records, index_data
        )

        self._candidates = self._sample(sampled_records, N_SAMPLED_RECORD_PAIRS)
        examples_to_index = self.candidates.copy()

        if index_include:
            examples_to_index += index_include

        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates: TrainingExamples) -> None:
        blocker = self.block_learner.blocker

        records = core.unique((record for pair in candidates for record in pair))

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in records}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(records)

    @overload
    def _sample(self, data: DataInt, sample_size: int) -> TrainingExamples:
        ...

    @overload
    def _sample(self, data: DataStr, sample_size: int) -> TrainingExamples:
        ...

    def _sample(self, data, sample_size):
        sample_indices = self._sample_indices(
            sample_size, len(data) * (len(data) - 1) // 2
        )

        sample = [(data[id_1], data[id_2]) for id_1, id_2 in sample_indices]

        return sample


class RecordLinkBlockLearner(BlockLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()

        N_SAMPLED_RECORDS = 4000
        N_SAMPLED_RECORD_PAIRS = 10000

        sampled_records_1 = sample_records(data_1, N_SAMPLED_RECORDS)
        index_data = sample_records(data_2, 50000)
        sampled_records_2 = sample_records(index_data, N_SAMPLED_RECORDS)

        preds = _filter_canopy_predicates(candidate_predicates, canopies=False)
        self.block_learner = training.RecordLinkBlockLearner(
            preds, sampled_records_1, sampled_records_2, index_data
        )

        self._candidates = self._sample(
            sampled_records_1, sampled_records_2, N_SAMPLED_RECORD_PAIRS
        )

        examples_to_index = self.candidates.copy()
        if index_include:
            examples_to_index += index_include
        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates: TrainingExamples) -> None:
        blocker = self.block_learner.blocker

        A_full, B_full = zip(*candidates)
        A = core.unique(A_full)
        B = core.unique(B_full)

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in B}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(A, B)

    @overload
    def _sample(
        self, data_1: DataInt, data_2: DataInt, sample_size: int
    ) -> TrainingExamples:
        ...

    @overload
    def _sample(
        self, data_1: DataStr, data_2: DataStr, sample_size: int
    ) -> TrainingExamples:
        ...

    def _sample(self, data_1, data_2, sample_size):
        sample_indices = self._sample_indices(sample_size, len(data_1) * len(data_2))

        sample = [(data_1[id_1], data_2[id_2]) for id_1, id_2 in sample_indices]

        return sample


class DisagreementLearner(HasCandidates):
    matcher: MatchLearner
    blocker: BlockLearner

    def __init__(self) -> None:
        self.y: numpy.typing.NDArray[numpy.int_] = numpy.array([])
        self.pairs: TrainingExamples = []

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        prob_l = [learner.candidate_scores() for learner in self._learners]
        probs = numpy.concatenate(prob_l, axis=1)

        # where do the classifers disagree?
        disagreement = numpy.std(probs > 0.5, axis=1).astype(bool)

        if disagreement.any():
            conflicts = disagreement.nonzero()[0]
            target = numpy.random.uniform(size=1)
            uncertain_index = conflicts[numpy.argmax(probs[conflicts][:, 0] - target)]
        else:
            uncertain_index = numpy.std(probs, axis=1).argmax()

        logger.debug(
            "Classifier: %.2f, Covered: %s",
            probs[uncertain_index][0],
            bool(probs[uncertain_index][1]),
        )

        uncertain_pair: TrainingExample = self.candidates[uncertain_index]
        self._remove(uncertain_index)
        return uncertain_pair

    @property
    def _learners(self) -> tuple[Learner, ...]:
        return (self.matcher, self.blocker)

    def _remove(self, index: int) -> None:
        self._candidates.pop(index)
        for learner in self._learners:
            learner.remove(index)

    def mark(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        self.y = numpy.concatenate([self.y, y])
        self.pairs.extend(pairs)
        for learner in self._learners:
            learner.fit(self.pairs, self.y)

    def learn_predicates(
        self, recall: float, index_predicates: bool
    ) -> tuple[Predicate, ...]:
        dupes = [pair for label, pair in zip(self.y, self.pairs) if label]
        return self.blocker.learn_predicates(
            dupes, recall=recall, index_predicates=index_predicates
        )


class DedupeDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        featurizer: FeaturizerFunction,
        data: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()
        data = core.index(data)

        random_pair = (
            random.choice(list(data.values())),
            random.choice(list(data.values())),
        )
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = DedupeBlockLearner(candidate_predicates, data, index_include)

        self._candidates = self.blocker.candidates.copy()

        self.matcher = MatchLearner(featurizer, self.candidates)

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


class RecordLinkDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        featurizer: FeaturizerFunction,
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()
        data_1 = core.index(data_1)

        offset = len(data_1)
        data_2 = core.index(data_2, offset)

        random_pair = (
            random.choice(list(data_1.values())),
            random.choice(list(data_2.values())),
        )
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = RecordLinkBlockLearner(
            candidate_predicates, data_1, data_2, index_include
        )
        self._candidates = self.blocker.candidates.copy()

        self.matcher = MatchLearner(featurizer, self.candidates)

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


def sample_records(data: Mapping, sample_size: int) -> dict:
    keys = data.keys()
    if len(data) > sample_size:
        keys = random.sample(tuple(keys), sample_size)  # type: ignore[assignment]
    # Always make a copy to avoid surprises
    return {k: data[k] for k in keys}
