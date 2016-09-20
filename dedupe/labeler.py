import numpy
import rlr
import random

class ActiveLearner(rlr.RegularizedLogisticRegression):
    def __init__(self, data_model, candidates):
        super().__init__()
        
        self.data_model = data_model
        self.candidates = candidates

        self.distances = self.transform(candidates)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])
        self.fit_transform([exact_match, exact_match, random_pair],
                           [1, 1, 0],
                           [1, 1, 1])

        
    def transform(self, pairs):
        return self.data_model.distances(pairs)

    def fit(self, X, y, weights):
        self.y = y
        self.X = X
        self.case_weights = weights

        super().fit(X, y, weights, cv=False)

    def fit_transform(self, pairs, y, weights):
        self.fit(self.transform(pairs), y, weights)

    def get(self):
        target_uncertainty = self._bias()

        probabilities = self.predict_proba(self.distances)
        uncertain_index = numpy.abs(target_uncertainty - probabilities).argmin()

        self.distances = numpy.delete(self.distances, uncertain_index, axis=0)

        uncertain_pairs = [self.candidates.pop(uncertain_index)]

        return uncertain_pairs

    def mark(self, pair, y, case_weight):
        
        self.y = numpy.append(self.y, y)
        self.X = numpy.vstack([self.X, self.transform([pair])])
        self.case_weights = numpy.append(self.case_weights, case_weight)

        super().fit(self.X, self.y, self.case_weights, cv=0)

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
