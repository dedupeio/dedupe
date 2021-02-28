import random
from abc import ABC, abstractmethod
import logging

import numpy
import rlr
from typing import List
from typing_extensions import Protocol

import dedupe.sampling as sampling
import dedupe.core as core
import dedupe.training as training
import dedupe.datamodel as datamodel
from dedupe._typing import TrainingExample

logger = logging.getLogger(__name__)


class ActiveLearner(ABC):

    @abstractmethod
    def transform(self) -> None:
        pass

    @abstractmethod
    def pop(self) -> TrainingExample:
        pass

    @abstractmethod
    def mark(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class HasDataModel(Protocol):

    data_model: datamodel.DataModel


class DedupeSampler(object):

    def _sample(self: HasDataModel, data, blocked_proportion, sample_size) -> List[TrainingExample]:
        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(self.data_model.predicates(index_predicates=False))

        data = sampling.randomDeque(data)
        blocked_sample_keys = sampling.dedupeBlockedSample(blocked_sample_size,
                                                           predicates,
                                                           data)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = set(core.randomPairs(len(data),
                                                  random_sample_size))
        data = dict(data)

        return [(data[k1], data[k2])
                for k1, k2
                in blocked_sample_keys | random_sample_keys]


class RecordLinkSampler(object):

    def _sample(self: HasDataModel, data_1, data_2, blocked_proportion, sample_size) -> List[TrainingExample]:
        offset = len(data_1)

        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(self.data_model.predicates(index_predicates=False))

        deque_1 = sampling.randomDeque(data_1)
        deque_2 = sampling.randomDeque(data_2)

        blocked_sample_keys = sampling.linkBlockedSample(blocked_sample_size,
                                                         predicates,
                                                         deque_1,
                                                         deque_2)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = core.randomPairsMatch(len(deque_1),
                                                   len(deque_2),
                                                   random_sample_size)

        unique_random_sample_keys = {(a, b + offset)
                                     for a, b in random_sample_keys}

        return [(data_1[k1], data_2[k2])
                for k1, k2
                in blocked_sample_keys | unique_random_sample_keys]


class RLRLearner(ActiveLearner, rlr.RegularizedLogisticRegression):
    def __init__(self, data_model):
        super().__init__(alpha=1)
        self.data_model = data_model
        self._candidates: List[TrainingExample]

    @property
    def candidates(self) -> List[TrainingExample]:
        return self._candidates

    @candidates.setter
    def candidates(self, new_candidates):
        self._candidates = new_candidates

        self.distances = self.transform(self._candidates)

        random_pair = random.choice(self._candidates)
        exact_match = (random_pair[0], random_pair[0])
        self.fit_transform([exact_match, random_pair],
                           [1, 0])

    def transform(self, pairs):
        return self.data_model.distances(pairs)

    def fit(self, X, y):

        self.y = numpy.array(y)
        self.X = X

        super().fit(self.X, self.y, cv=False)

    def fit_transform(self, pairs, y):
        self.fit(self.transform(pairs), y)

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        target_uncertainty = self._bias()

        probabilities = self.candidate_scores()

        distance_to_target = numpy.abs(target_uncertainty - probabilities)
        uncertain_index = distance_to_target.argmin()

        self.distances = numpy.delete(self.distances, uncertain_index, axis=0)

        uncertain_pair = self.candidates.pop(uncertain_index)

        return uncertain_pair

    def _remove(self, index):
        self.distances = numpy.delete(self.distances, index, axis=0)

    def mark(self, pairs, y):

        self.y = numpy.concatenate([self.y, y])
        self.X = numpy.vstack([self.X, self.transform(pairs)])

        self.fit(self.X, self.y)

    def _bias(self):
        positive = numpy.sum(self.y == 1)
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

    def candidate_scores(self):
        return self.predict_proba(self.distances)

    def __len__(self):
        return len(self.candidates)


class DedupeRLRLearner(DedupeSampler, RLRLearner):
    def __init__(self, data_model, data, blocked_proportion, sample_size):
        super().__init__(data_model)
        self.candidates = self._sample(data, blocked_proportion, sample_size)


class RecordLinkRLRLearner(RecordLinkSampler, RLRLearner):
    def __init__(self, data_model, data_1, data_2, blocked_proportion, sample_size):
        super.__init__(data_model)
        self.candidates = self._sample(data_1, data_2, blocked_proportion, sample_size)


class BlockLearner(object):

    def __init__(self, data_model, candidates, *args):
        self.data_model = data_model
        self.candidates = candidates

        self.current_predicates = ()

        self._cached_labels = None
        self._old_dupes = []

        self.block_learner: training.BlockLearner

    def fit_transform(self, pairs, y):
        dupes = [pair for label, pair in zip(y, pairs) if label]

        new_dupes = [pair for pair in dupes if pair not in self._old_dupes]
        new_uncovered = (not all(self.predict(new_dupes)))

        if new_uncovered:
            self.current_predicates = self.block_learner.learn(dupes,
                                                               recall=1.0)
            self._cached_labels = None
            self._old_dupes = dupes

    def candidate_scores(self):
        if self._cached_labels is None:
            labels = self.predict(self.candidates)
            self._cached_labels = numpy.array(labels).reshape(-1, 1)

        return self._cached_labels

    def predict(self, candidates):
        labels = []
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

    def _remove(self, index):
        if self._cached_labels is not None:
            self._cached_labels = numpy.delete(self._cached_labels,
                                               index,
                                               axis=0)


class DedupeBlockLearner(BlockLearner):

    def __init__(self, data_model,
                 candidates,
                 data,
                 original_length,
                 index_include):
        super().__init__(data_model, candidates)

        index_data = Sample(data, 50000, original_length)
        sampled_records = Sample(index_data, 5000, original_length)
        preds = self.data_model.predicates()

        self.block_learner = training.DedupeBlockLearner(preds,
                                                         sampled_records,
                                                         index_data)

        examples_to_index = candidates.copy()
        if index_include:
            examples_to_index += index_include

        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates):

        blocker = self.block_learner.blocker

        records = core.unique((record for pair in candidates for record in pair))

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in records}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(records)


class RecordLinkBlockLearner(BlockLearner):

    def __init__(self,
                 data_model,
                 candidates,
                 data_1,
                 data_2,
                 original_length_1,
                 original_length_2,
                 index_include):

        super().__init__(data_model, candidates)

        sampled_records_1 = Sample(data_1, 600, original_length_1)
        index_data = Sample(data_2, 50000, original_length_2)
        sampled_records_2 = Sample(index_data, 600, original_length_2)

        preds = self.data_model.predicates(canopies=False)

        self.block_learner = training.RecordLinkBlockLearner(preds,
                                                             sampled_records_1,
                                                             sampled_records_2,
                                                             index_data)

        examples_to_index = candidates.copy()
        if index_include:
            examples_to_index += index_include

        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates):

        blocker = self.block_learner.blocker

        A, B = zip(*candidates)
        A = core.unique(A)
        B = core.unique(B)

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in B}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(A, B)


class DisagreementLearner(ActiveLearner):

    classifier: RLRLearner
    blocker: BlockLearner
    candidates: List[TrainingExample]

    def _common_init(self):

        self.learners = (self.classifier, self.blocker)
        self.y = numpy.array([])
        self.pairs = []

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        probs_l = []
        for learner in self.learners:
            probabilities = learner.candidate_scores()
            probs_l.append(probabilities)

        probs = numpy.concatenate(probs_l, axis=1)

        # where do the classifers disagree?
        disagreement = numpy.std(probs > 0.5, axis=1).astype(bool)

        if disagreement.any():
            conflicts = disagreement.nonzero()[0]  # type: ignore
            target = numpy.random.uniform(size=1)
            uncertain_index = conflicts[numpy.argmax(probs[conflicts][:, 0] - target)]
        else:
            uncertain_index = numpy.std(probs, axis=1).argmax()

        logger.debug("Classifier: %.2f, Covered: %s",
                     probs[uncertain_index][0],
                     bool(probs[uncertain_index][1]))

        uncertain_pair = self.candidates.pop(uncertain_index)

        for learner in self.learners:
            learner._remove(uncertain_index)

        return uncertain_pair

    def mark(self, pairs, y):

        self.y = numpy.concatenate([self.y, y])
        self.pairs.extend(pairs)

        for learner in self.learners:
            learner.fit_transform(self.pairs, self.y)

    def __len__(self):
        return len(self.candidates)

    def transform(self):
        pass

    def learn_predicates(self, recall, index_predicates):
        dupes = [pair for label, pair in zip(self.y, self.pairs) if label]

        if not index_predicates:
            old_preds = self.blocker.block_learner.blocker.predicates.copy()

            no_index_predicates = [pred for pred in old_preds
                                   if not hasattr(pred, 'index')]
            self.blocker.block_learner.blocker.predicates = no_index_predicates

            learned_preds = self.blocker.block_learner.learn(dupes,
                                                             recall=recall,
                                                             candidate_types='random forest')

            self.blocker.block_learner.blocker.predicates = old_preds

        else:
            learned_preds = self.blocker.block_learner.learn(dupes,
                                                             recall=recall,
                                                             candidate_types='random forest')

        return learned_preds


class DedupeDisagreementLearner(DedupeSampler, DisagreementLearner):

    def __init__(self,
                 data_model,
                 data,
                 blocked_proportion,
                 sample_size,
                 original_length,
                 index_include):

        self.data_model = data_model

        data = core.index(data)

        self.candidates = self._sample(data, blocked_proportion, sample_size)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = DedupeBlockLearner(data_model,
                                          self.candidates,
                                          data,
                                          original_length,
                                          index_include)
        self.classifier = RLRLearner(self.data_model)
        self.classifier.candidates = self.candidates

        self._common_init()

        self.mark([exact_match] * 4 + [random_pair],
                  [1] * 4 + [0])


class RecordLinkDisagreementLearner(RecordLinkSampler, DisagreementLearner):

    def __init__(self,
                 data_model,
                 data_1,
                 data_2,
                 blocked_proportion,
                 sample_size,
                 original_length_1,
                 original_length_2,
                 index_include):

        self.data_model = data_model

        data_1 = core.index(data_1)

        offset = len(data_1)
        data_2 = core.index(data_2, offset)

        self.candidates = self._sample(data_1,
                                       data_2,
                                       blocked_proportion,
                                       sample_size)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = RecordLinkBlockLearner(data_model,
                                              self.candidates,
                                              data_1,
                                              data_2,
                                              original_length_1,
                                              original_length_2,
                                              index_include)
        self.classifier = RLRLearner(self.data_model)
        self.classifier.candidates = self.candidates

        self._common_init()

        self.mark([exact_match] * 4 + [random_pair],
                  [1] * 4 + [0])


class Sample(dict):

    def __init__(self, d, sample_size, original_length):
        if len(d) <= sample_size:
            super().__init__(d)
        else:
            sample = random.sample(d.keys(), sample_size)
            super().__init__({k: d[k] for k in sample})
        if original_length is None:
            self.original_length = len(d)
        else:
            self.original_length = original_length
