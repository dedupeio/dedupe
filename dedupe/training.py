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


def activeLearning(candidates,
                   data_model,
                   labelPairFunction,
                   training_data,
                   training_pairs=None):
    """
    Ask the user to label the record pair we are most uncertain of. Train the
    data model, and update our uncertainty. Repeat until user tells us she is
    finished.
    """

    fields = [field for field in data_model['fields']
              if data_model['fields'][field]['type'] not in ('Missing Data',
                                                             'Interaction')]


    duplicates = []
    nonduplicates = []

    if training_pairs:
        nonduplicates.extend(training_pairs[0])
        duplicates.extend(training_pairs[1])


    if training_data.shape[0] == 0 :
        rand_int = random.randint(0, len(candidates))
        exact_match = candidates[rand_int]
        training_data = addTrainingData({1:[exact_match]*2,
                                         0:[]},
                                        data_model,
                                        training_data)

    data_model = core.trainModel(training_data, data_model, .1)


    finished = False

    import time
    t_train = time.time()
    field_distances = core.fieldDistances(candidates, data_model)
    logging.info('calculated fieldDistances in %s seconds',
                 str(time.time() - t_train))

    seen_indices = set()

    while finished == False:
        logging.info('finding the next uncertain pair ...')
        uncertain_indices = findUncertainPairs(field_distances,
                                               data_model,
                                               (len(duplicates)/
                                                (len(nonduplicates)+1.0)))

        for uncertain_index in uncertain_indices:
            if uncertain_index not in seen_indices:
                seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [candidates[uncertain_index]]

        (labeled_pairs, finished) = labelPairFunction(uncertain_pairs, fields)

        nonduplicates.extend(labeled_pairs[0])
        duplicates.extend(labeled_pairs[1])

        training_data = addTrainingData(labeled_pairs, data_model, training_data)

        if len(training_data) > 0:

            data_model = core.trainModel(training_data, data_model, .1)
        else:
            raise ValueError('No training pairs given')

    training_pairs = {0: nonduplicates, 1: duplicates}

    return (training_data, training_pairs, data_model)


def addTrainingData(labeled_pairs, data_model, training_data=[]):
    """
    Appends training data to the training data collection.
    """

    fields = data_model['fields']

    examples = [record_pair for example in labeled_pairs.values()
                for record_pair in example]

    new_training_data = numpy.empty(len(examples),
                                    dtype=training_data.dtype)

    new_training_data['label'] = [0] * len(labeled_pairs[0]) + [1] * len(labeled_pairs[1])
    new_training_data['distances'] = core.fieldDistances(examples, data_model)


    training_data = numpy.append(training_data, new_training_data)


    return training_data


def consoleLabel(uncertain_pairs, fields):
    '''Command line interface for presenting and labeling training pairs by the user'''
    duplicates = []
    nonduplicates = []
    finished = False


    for record_pair in uncertain_pairs:
        label = ''

        for pair in record_pair:
            for field in fields:
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

        if label == 'y':
            duplicates.append(record_pair)
        elif label == 'n':
            nonduplicates.append(record_pair)
        elif label == 'f':
            sys.stderr.write('Finished labeling\n')
            finished = True
            break
        elif label != 'u':
            sys.stderr.write('Nonvalid response\n')
            raise

    return ({0: nonduplicates, 1: duplicates}, finished)


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

    


