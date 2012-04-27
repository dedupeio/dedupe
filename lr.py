# An implementation of Logistic Regression algorithm in python.
# The class use sparse representation of features.
# Author: Xiance Si (sixiance<at>gmail.com)

import math

class LogisticRegression:
    # Initialize member variables. We have two member variables
    # 1) weight: a dict obejct storing the weight of all features.
    # 2) bias: a float value of the bias value.
    def __init__(self):
        self.rate = 0.01
        self.weight = {}
        return
    # data is a list of [label, feature]. label is an integer,
    # 1 for positive instance, 0 for negative instance. feature is
    # a dict object, the key is feature name, the value is feature
    # weight.
    #
    # n is the number of training iterations.
    #
    # We use online update formula to train the model.
    def train(self, data, n):
        for i in range(n):
            for [label, feature] in data:
                predicted = self.classify(feature)
                for f,v in feature.iteritems():
                    if f not in self.weight:
                        self.weight[f] = 0
                    update = (label - predicted) * v
                    self.weight[f] = self.weight[f] + self.rate * update
            print 'iteration', i, 'done'
        return
    # feature is a dict object, the key is feature name, the value
    # is feature weight. Return value is the probability of being
    # a positive instance.
    def classify(self, feature):
        logit = 0
        for f,v in feature.iteritems():
            coef = 0
            if f in self.weight:
                coef = self.weight[f]
            logit += coef * v
        return 1.0 / (1.0 + math.exp(-logit))
