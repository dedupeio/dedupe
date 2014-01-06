#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

from itertools import combinations, islice
import blocking
import core
import numpy
import logging
import random
import sys

def findUncertainPairs(field_distances, data_model, bias=0.5):
    """
    Given a set of field distances and a data model return the
    indices of the record pairs in order of uncertainty. For example,
    the first indices corresponds to the record pair where we have the
    least certainty whether the pair are duplicates or distinct.
    """

    probability = core.scorePairs(field_distances, data_model)

    p_max = (1.0 - bias)
    logging.info(p_max)

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

    def getUncertainPair(self, data_model, dupe_ratio) :
        uncertain_indices = findUncertainPairs(self.field_distances,
                                               data_model,
                                               dupe_ratio)

        for uncertain_index in uncertain_indices:
            if uncertain_index not in self.seen_indices:
                self.seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [self.candidates[uncertain_index]]

        return uncertain_pairs

def consoleLabel(deduper):
    '''Command line interface for presenting and labeling training pairs by the user'''


    finished = False

    while not finished :
        uncertain_pairs = deduper.getUncertainPair()

        labels = {'distinct' : [], 'match' : []}

        for record_pair in uncertain_pairs:
            label = ''
            labeled = False

            for pair in record_pair:
                for field in deduper.data_model.comparison_fields:
                    line = "%s : %s\n" % (field, pair[field])
                    sys.stderr.write(line)
                sys.stderr.write('\n')

            sys.stderr.write('Do these records refer to the same thing?\n')

            valid_response = False
            while not valid_response:
                sys.stderr.write('(y)es / (n)o / (u)nsure / (f)inished\n')
                label = sys.stdin.readline().strip()
                if label in ['y', 'n', 'u', 'f']:
                    valid_response = True

            if label == 'y' :
                labels['match'].append(record_pair)
                labeled = True
            elif label == 'n' :
                labels['distinct'].append(record_pair)
                labeled = True
            elif label == 'f':
                sys.stderr.write('Finished labeling\n')
                finished = True
            elif label != 'u':
                sys.stderr.write('Nonvalid response\n')
                raise

        if labeled :
            deduper.markPairs(labels)

    deduper.trainClassifier()
    deduper.trainBlocker()

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

    



