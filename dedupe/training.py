#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

from itertools import combinations, islice
import blocking
import core
import numpy
import logging
import random

logger = logging.getLogger(__name__)

def findUncertainPairs(field_distances, data_model, bias=0.5):
    """
    Given a set of field distances and a data model return the
    indices of the record pairs in order of uncertainty. For example,
    the first indices corresponds to the record pair where we have the
    least certainty whether the pair are duplicates or distinct.
    """

    probability = core.scorePairs(field_distances, data_model)

    p_max = (1.0 - bias)
    logger.info(p_max)

    informativity = numpy.copy(probability)
    informativity[probability < p_max] /= p_max
    informativity[probability >= p_max] = (1 - probability[probability >= p_max])/(1-p_max)


    return numpy.argsort(-informativity)


class ActiveLearning(object) :
    """
    Ask the user to label the record pair we are most uncertain of. Train the
    data model, and update our uncertainty. Repeat until user tells us she is
    finished.
    """
    def __init__(self, candidates, data_model) :

        self.candidates = candidates
        self.field_distances = core.fieldDistances(self.candidates, data_model)
        self.seen_indices = set()

    def uncertainPairs(self, data_model, dupe_ratio) :
        uncertain_indices = findUncertainPairs(self.field_distances,
                                               data_model,
                                               dupe_ratio)

        for uncertain_index in uncertain_indices:
            if uncertain_index not in self.seen_indices:
                self.seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [self.candidates[uncertain_index]]

        return uncertain_pairs


def semiSupervisedNonDuplicates(data_sample,
                                data_model,
                                nonduplicate_confidence_threshold=.7,
                                sample_size=2000):

    confidence = 1 - nonduplicate_confidence_threshold

    def distinctPairs() :
        data_slice = data_sample[0:sample_size]
        pair_distance = core.fieldDistances(data_slice, data_model)
        scores = core.scorePairs(pair_distance, data_model)

        sample_n = 0
        for score, pair in zip(scores, data_sample) :
            if score < confidence :
                yield pair
                sample_n += 1

        if sample_n < sample_size and len(data_sample) > sample_size :
            for pair in data_sample[sample_size:] :
                pair_distance = core.fieldDistances([pair], data_model)
                score = core.scorePairs(pair_distance, data_model)
                
                if score < confidence :
                    yield (pair)

    return islice(distinctPairs(), 0, sample_size)

    



