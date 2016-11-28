from __future__ import division
from future.utils import with_metaclass

import random
from abc import ABCMeta, abstractmethod

import numpy
import rlr

import dedupe.sampling as sampling
import dedupe.core as core

class ActiveLearner(with_metaclass(ABCMeta)):

    @abstractmethod
    def transform():
        pass

    @abstractmethod
    def get():
        pass

    @abstractmethod
    def mark():
        pass

    @abstractmethod
    def __len__(self):
        pass

    def sample_combo(self, data, blocked_proportion, sample_size):
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

        self.candidates = [(data[k1], data[k2])
                           for k1, k2
                           in blocked_sample_keys | random_sample_keys]

        self.distances = self.transform(self.candidates)
        

    def sample_product(self, data_1, data_2, blocked_proportion, sample_size):
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

        random_sample_keys = {(a, b + offset)
                              for a, b in random_sample_keys}

        self.candidates = [(data_1[k1], data_2[k2])
                           for k1, k2
                           in blocked_sample_keys | random_sample_keys]

        self.distances = self.transform(self.candidates)

            

class RLRLearner(ActiveLearner, rlr.RegularizedLogisticRegression):
    def __init__(self, data_model):
        super(RLRLearner, self).__init__(alpha=1)
        
        self.data_model = data_model
        
    def transform(self, pairs):
        return self.data_model.distances(pairs)

    def fit(self, X, y):
        self.y = numpy.array(y)
        self.X = X

        super(RLRLearner, self).fit(self.X, self.y, cv=False)

    def fit_transform(self, pairs, y):
        self.fit(self.transform(pairs), y)

    def get(self):
        target_uncertainty = self._bias()

        probabilities = self.predict_proba(self.distances)
        uncertain_index = numpy.abs(target_uncertainty - probabilities).argmin()

        self.distances = numpy.delete(self.distances, uncertain_index, axis=0)

        uncertain_pair = self.candidates.pop(uncertain_index)

        return [uncertain_pair]

    def mark(self, pairs, y):

        self.y = numpy.concatenate([self.y, y])
        self.X = numpy.vstack([self.X, self.transform(pairs)])

        self.fit(self.X, self.y)

    def _bias(self):
        positive = numpy.sum(self.y == 1)
        n_examples = len(self.y)

        bias = 1 - (positive/n_examples if positive else 0)

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

    def __len__(self):
        return len(self.candidates)

    def _init_rlr(self):
        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])
        self.fit_transform([exact_match, random_pair],
                           [1, 0])
        
    def sample_combo(self, *args):
        super(RLRLearner, self).sample_combo(*args)
        self._init_rlr()

    def sample_product(self, *args):
        super(RLRLearner, self).sample_product(*args)
        self._init_rlr()
        
