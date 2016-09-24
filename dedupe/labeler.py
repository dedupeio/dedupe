from __future__ import division
from future.utils import with_metaclass

import numpy
import rlr
import random
from abc import ABCMeta, abstractmethod

class ActiveLearner(with_metaclass(ABCMeta)):

    @abstractmethod
    def __init__():
        pass

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

class RLRLearner(ActiveLearner, rlr.RegularizedLogisticRegression):
    def __init__(self, data_model, candidates):
        super(ActiveLearner, self).__init__()
        
        self.data_model = data_model
        self.candidates = candidates

        self.distances = self.transform(candidates)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])
        self.fit_transform([exact_match, random_pair],
                           [1, 0])

        
    def transform(self, pairs):
        return self.data_model.distances(pairs)

    def fit(self, X, y):
        self.y = numpy.array(y)
        self.X = X

        super(ActiveLearner, self).fit(self.X, self.y, cv=False)

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

        super(ActiveLearner, self).fit(self.X, self.y, cv=0)

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
