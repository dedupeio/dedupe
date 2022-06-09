from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy
import numpy.typing
import sklearn.linear_model

import dedupe.core as core
import dedupe.training as training

if TYPE_CHECKING:
    from typing import Dict, Iterable, Mapping

    from dedupe._typing import Data, Labels, LabelsLike
    from dedupe._typing import RecordDictPair as TrainingExample
    from dedupe._typing import RecordDictPairs as TrainingExamples
    from dedupe._typing import RecordIDPair
    from dedupe.datamodel import DataModel
    from dedupe.predicates import Predicate


logger = logging.getLogger(__name__)


class Learner(ABC):
    candidates: TrainingExamples

    @abstractmethod
    def fit_transform(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        pass

    @abstractmethod
    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        pass

    @abstractmethod
    def _remove(self, index: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.candidates)


class ActiveLearner(Learner):
    @abstractmethod
    def pop(self) -> TrainingExample:
        pass

    @abstractmethod
    def mark(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        pass


class RLRLearner(sklearn.linear_model.LogisticRegression, ActiveLearner):
    def __init__(self, data_model: DataModel):
        super().__init__()
        self.data_model = data_model
        self._candidates: TrainingExamples = []

    @property  # type: ignore[override]
    def candidates(self) -> TrainingExamples:  # type: ignore[override]
        return self._candidates

    @candidates.setter
    def candidates(self, new_candidates: TrainingExamples) -> None:
        self._candidates = new_candidates

        self.distances = self.transform(self._candidates)

        random_pair = random.choice(self._candidates)
        exact_match = (random_pair[0], random_pair[0])
        self.fit_transform([exact_match, random_pair], [1, 0])

    def transform(self, pairs: TrainingExamples) -> numpy.typing.NDArray[numpy.float_]:
        return self.data_model.distances(pairs)

    def fit(self, X: numpy.typing.NDArray[numpy.float_], y: LabelsLike) -> None:

        self.y: numpy.typing.NDArray[numpy.int_] = numpy.array(y)
        self.X = X

        super().fit(self.X, self.y)

    def fit_transform(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        self.fit(self.transform(pairs), y)

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        target_uncertainty = self._bias()

        probabilities = self.candidate_scores()

        distance_to_target = numpy.abs(target_uncertainty - probabilities)
        uncertain_index = distance_to_target.argmin()

        self.distances = numpy.delete(self.distances, uncertain_index, axis=0)

        uncertain_pair: TrainingExample = self.candidates.pop(uncertain_index)

        return uncertain_pair

    def _remove(self, index: int) -> None:
        self.distances = numpy.delete(self.distances, index, axis=0)

    def mark(self, pairs: TrainingExamples, y: LabelsLike) -> None:

        self.y = numpy.concatenate([self.y, y])  # type: ignore[arg-type]
        self.X = numpy.vstack([self.X, self.transform(pairs)])

        self.fit(self.X, self.y)

    def _bias(self) -> float:
        positive: int = numpy.sum(self.y == 1)
        n_examples = len(self.y)

        bias = 1 - (positive / n_examples if positive else 0)

        # When we have just a few examples we are okay with getting
        # examples where the model strongly believes the example is
        # going to be positive or negative. As we get more examples,
        # prefer to ask for labels of examples the model is more
        # uncertain of.
        uncertainty_weight = min(positive, n_examples - positive)
        bias_weight = 10

        weighted_bias = 0.5 * uncertainty_weight + bias * bias_weight
        weighted_bias /= uncertainty_weight + bias_weight

        return weighted_bias

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        scores: numpy.typing.NDArray[numpy.float_] = self.predict_proba(self.distances)[
            :, 1
        ].reshape(-1, 1)
        return scores


class BlockLearner(Learner):
    candidates: TrainingExamples

    def __init__(self, data_model: DataModel, *args):
        self.data_model = data_model

        self.current_predicates: tuple[Predicate, ...] = ()

        self._cached_labels: numpy.typing.NDArray[numpy.float_] | None = None
        self._old_dupes: TrainingExamples = []

        self.block_learner: training.BlockLearner

    def fit_transform(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        dupes = [pair for label, pair in zip(y, pairs) if label]

        new_dupes = [pair for pair in dupes if pair not in self._old_dupes]
        new_uncovered = not all(self.predict(new_dupes))

        if new_uncovered:
            self.current_predicates = self.block_learner.learn(dupes, recall=1.0)
            self._cached_labels = None
            self._old_dupes = dupes

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        if self._cached_labels is None:
            labels = self.predict(self.candidates)
            self._cached_labels = numpy.array(labels).reshape(-1, 1)

        return self._cached_labels

    def predict(self, candidates: TrainingExamples) -> Labels:
        labels: Labels = []
        for record_1, record_2 in candidates:

            for predicate in self.current_predicates:
                keys = predicate(record_2, target=True)
                if keys:
                    if set(predicate(record_1)) & set(keys):
                        labels.append(1)
                        break
            else:
                labels.append(0)

        return labels

    def _remove(self, index: int) -> None:
        if self._cached_labels is not None:
            self._cached_labels = numpy.delete(self._cached_labels, index, axis=0)

    def _sample_indices(self, sample_size: int) -> Iterable[RecordIDPair]:

        weights: Dict[RecordIDPair, float] = {}
        for predicate, covered in self.block_learner.comparison_cover.items():
            # each predicate gets to vote for every record pair it covers. the
            # strength of that vote is in inverse proportion to the number of
            # records the predicate covers.
            #
            # if a predicate only covers a few record pairs, the value of
            # the vote it puts on those few pairs will be worth more than
            # a predicate that covers almost all the record pairs
            weight: float = 1 / len(covered)
            for pair in covered:
                weights[pair] = weights.get(pair, 0.0) + weight

        sample_ids: Iterable[RecordIDPair]
        if sample_size < len(weights):
            # consider using a reservoir sampling strategy, which would
            # be more memory efficient and probably about as fast
            normalized_weights = numpy.fromiter(weights.values(), dtype=float) / sum(
                weights.values()
            )
            rng = numpy.random.default_rng()
            sample_indices = rng.choice(
                len(weights), size=sample_size, replace=False, p=normalized_weights
            )
            keys = list(weights.keys())
            sample_ids = ((keys[i][0], keys[i][1]) for i in sample_indices)
        else:
            sample_ids = weights.keys()

        return sample_ids


class DedupeBlockLearner(BlockLearner):
    def __init__(
        self,
        data_model: DataModel,
        data: Data,
        index_include: TrainingExamples,
    ):
        super().__init__(data_model)

        N_SAMPLED_RECORDS = 5000
        N_SAMPLED_RECORD_PAIRS = 10000

        index_data = sample_records(data, 50000)
        sampled_records = sample_records(index_data, N_SAMPLED_RECORDS)
        preds = self.data_model.predicates()

        self.block_learner = training.DedupeBlockLearner(
            preds, sampled_records, index_data
        )

        self.candidates = self._sample(sampled_records, N_SAMPLED_RECORD_PAIRS)
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

    def _sample(self, data: Data, sample_size: int) -> TrainingExamples:

        sample_indices = self._sample_indices(sample_size)

        sample = [(data[id_1], data[id_2]) for id_1, id_2 in sample_indices]

        return sample


class RecordLinkBlockLearner(BlockLearner):
    def __init__(
        self,
        data_model: DataModel,
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):

        super().__init__(data_model)

        N_SAMPLED_RECORDS = 1000
        N_SAMPLED_RECORD_PAIRS = 5000

        sampled_records_1 = sample_records(data_1, N_SAMPLED_RECORDS)
        index_data = sample_records(data_2, 50000)
        sampled_records_2 = sample_records(index_data, N_SAMPLED_RECORDS)

        preds = self.data_model.predicates(canopies=False)

        self.block_learner = training.RecordLinkBlockLearner(
            preds, sampled_records_1, sampled_records_2, index_data
        )

        self.candidates = self._sample(
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

    def _sample(self, data_1: Data, data_2: Data, sample_size: int) -> TrainingExamples:

        sample_indices = self._sample_indices(sample_size)

        sample = [(data_1[id_1], data_2[id_2]) for id_1, id_2 in sample_indices]

        return sample


class DisagreementLearner(ActiveLearner):

    classifier: RLRLearner
    blocker: BlockLearner
    candidates: TrainingExamples

    def _common_init(self) -> None:

        self.learners: tuple[Learner, ...] = (self.classifier, self.blocker)
        self.y: numpy.typing.NDArray[numpy.int_] = numpy.array([])
        self.pairs: TrainingExamples = []

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        probs = self.candidate_scores()

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

        uncertain_pair: TrainingExample = self.candidates.pop(uncertain_index)

        self._remove(uncertain_index)

        return uncertain_pair

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float_]:
        probs_l = []
        for learner in self.learners:
            probabilities = learner.candidate_scores()
            probs_l.append(probabilities)

        return numpy.concatenate(probs_l, axis=1)

    def _remove(self, index: int) -> None:
        for learner in self.learners:
            learner._remove(index)

    def mark(self, pairs: TrainingExamples, y: LabelsLike) -> None:

        self.y = numpy.concatenate([self.y, y])  # type: ignore[arg-type]
        self.pairs.extend(pairs)

        self.fit_transform(self.pairs, self.y)

    def fit_transform(self, pairs: TrainingExamples, y: LabelsLike) -> None:

        for learner in self.learners:
            learner.fit_transform(pairs, y)

    def learn_predicates(
        self, recall: float, index_predicates: bool
    ) -> tuple[Predicate, ...]:

        learned_preds: tuple[Predicate, ...]
        dupes = [pair for label, pair in zip(self.y, self.pairs) if label]

        if not index_predicates:
            old_preds = self.blocker.block_learner.blocker.predicates.copy()  # type: ignore[attr-defined]

            no_index_predicates = [
                pred for pred in old_preds if not hasattr(pred, "index")
            ]
            self.blocker.block_learner.blocker.predicates = no_index_predicates

            learned_preds = self.blocker.block_learner.learn(
                dupes, recall=recall, candidate_types="random forest"
            )

            self.blocker.block_learner.blocker.predicates = old_preds

        else:
            learned_preds = self.blocker.block_learner.learn(
                dupes, recall=recall, candidate_types="random forest"
            )

        return learned_preds


class DedupeDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        data_model: DataModel,
        data: Data,
        index_include: TrainingExamples,
    ):

        self.data_model = data_model

        data = core.index(data)

        random_pair = (
            random.choice(list(data.values())),
            random.choice(list(data.values())),
        )
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = DedupeBlockLearner(data_model, data, index_include)

        self.candidates = self.blocker.candidates

        self.classifier = RLRLearner(self.data_model)
        self.classifier.candidates = self.candidates

        self._common_init()

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


class RecordLinkDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        data_model: DataModel,
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):

        self.data_model = data_model

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

        self.blocker = RecordLinkBlockLearner(data_model, data_1, data_2, index_include)
        self.candidates = self.blocker.candidates

        self.classifier = RLRLearner(self.data_model)
        self.classifier.candidates = self.candidates

        self._common_init()

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


def sample_records(data: Mapping, sample_size: int) -> Mapping:
    if len(data) <= sample_size:
        return data
    else:
        sample = random.sample(data.keys(), sample_size)
        return {k: data[k] for k in sample}
