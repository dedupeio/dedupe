# An implementation of Logistic Regression algorithm in python.
# The class use sparse representation of features.
# Author: Xiance Si (sixiance<at>gmail.com)

from random import shuffle
import math
from collections import defaultdict
import copy

class LogisticRegression:
    # Initialize member variables. We have two member variables
    # 1) weight: a dict obejct storing the weight of all features.
    # 2) bias: a float value of the bias value.
    def __init__(self):
        self.rate = 0.01
        self.weight = defaultdict(int)
        self.bias = 0
        self.alpha = 0.001
        return
    # data is a list of [label, feature]. label is an integer,
    # 1 for positive instance, 0 for negative instance. feature is
    # a dict object, the key is feature name, the value is feature
    # weight.
    #
    # n is the number of training iterations.
    #
    # We use online update formula to train the model.
    def train(self, data, epochs=100):
        shuffle(data)
        
        #start_epoch = epochs/2
        #tstart = start_epoch * len(data)
        
        #avg_weight = defaultdict(int)
        #avg_bias = 0

        for epoch in range(epochs):
            epoch_start = epoch * len(data) + 1
            for t, (label, feature) in enumerate(data, epoch_start):
                rate_t = self.rate/((1 + self.alpha * self.rate * t)**0.75)
                #rate_t = self.rate * (1-t/float((epochs * len(data))))

                predicted = 1. / (1. + math.exp(-self.classify(feature)))

                gradient = label - predicted
                for f,v in feature.iteritems():
                    weight_gradient = gradient * v - self.alpha * self.weight[f]
                    self.weight[f] += rate_t * weight_gradient

                    #if epoch > 0 and epoch > (start_epoch - 1) :
                    #    avg_weight[f] += (self.weight[f] - avg_weight[f])/(t-tstart)
                self.bias += rate_t * gradient

                #if epoch > 0 and epoch > (start_epoch -1 ) :
                #    avg_bias += (self.bias - avg_bias) /( t-tstart)            

            #if self.alpha == .01 :
            #    print "Bias: ", self.bias
            #    print "Average Bias: ", avg_bias

        #if epoch > 0 and epoch > (start_epoch -1 ) :
        #    self.weight = avg_weight
        #    self.bias = avg_bias
            


        return
    # feature is a dict object, the key is feature name, the value
    # is feature weight. Return value is the probability of being
    # a positive instance.
    def classify(self, feature):
        logit = self.bias
        for f,v in feature.iteritems():
            coef = 0
            if f in self.weight:
                coef = self.weight[f]
            logit += coef * v 
        return logit 

    def evaluateLearnRate(self, subset, rate) :
        self.rate = rate
        self.train(subset, 1)
        
        loss = 0
        for label, feature in subset :
            if label == 0 :
                loss += math.log(1 + math.exp(1 * self.classify(feature)))
            else :
                loss += math.log(1 + math.exp(-1 * self.classify(feature)))

        weight_norm = sum([v*v for v in self.weight.values()])
            
        cost = loss/float(len(subset)) + 0.5 * self.alpha * weight_norm

        return cost

    def determineLearnRate(self, subset) :
        return .01

    def kdetermineLearnRate(self, subset) :
        factor = 2.0
        low_rate = 1
        low_cost = self.evaluateLearnRate(subset, low_rate)
        high_rate = low_rate * factor
        high_cost = self.evaluateLearnRate(subset, high_rate)
        if low_cost < high_cost :
            while low_cost < high_cost :
                high_rate = low_rate
                high_cost = low_cost
                low_rate = high_rate/factor
                low_cost = self.evaluateLearnRate(subset, low_rate)
        elif high_cost < low_cost :
            while high_cost < low_cost :
                low_rate = high_rate
                low_cost = high_cost
                high_rate = low_rate * factor
                high_cost = self.evaluateLearnRate(subset, high_rate)

        self.rate = low_rate
        print "Learning Rate: ", self.rate, low_cost

        return
