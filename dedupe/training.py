#!/usr/bin/python
# -*- coding: utf-8 -*-

# provides functions for selecting a sample of training data

from itertools import combinations
import blocking
import core
import numpy
import logging


def findUncertainPairs(record_distances, data_model):
    """
    Given a set of record distances and a data model return the
    indices of the record pairs in order of uncertainty. For example,
    the first indices corresponds to the record pair where we have the
    least certainty whether the pair are duplicates or distinct.
    """

    probability = core.scorePairs(record_distances, data_model)

    uncertainties = (probability * numpy.log2(probability) 
                     + (1 - probability) * numpy.log2(1 - probability))

    return numpy.argsort(uncertainties)


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

    duplicates = []
    nonduplicates = []

    if training_pairs:
        nonduplicates.extend(training_pairs[0])
        duplicates.extend(training_pairs[1])

    finished = False

    import time
    t_train = time.time()
    record_distances = core.recordDistances(candidates, data_model)
    logging.info('calculated recordDistances in %s seconds',
                 str(time.time() - t_train))

    seen_indices = set()

    while finished == False:
        logging.info('finding the next uncertain pair ...')
        uncertain_indices = findUncertainPairs(record_distances, data_model)

        for uncertain_index in uncertain_indices:
            if uncertain_index not in seen_indices:
                seen_indices.add(uncertain_index)
                break

        uncertain_pairs = [(candidates[uncertain_index][0][1],
                           candidates[uncertain_index][1][1])]

        (labeled_pairs, finished) = labelPairFunction(uncertain_pairs, data_model)

        nonduplicates.extend(labeled_pairs[0])
        duplicates.extend(labeled_pairs[1])

        training_data = addTrainingData(labeled_pairs, data_model, training_data)
        if len(training_data) > 0:
            data_model = core.trainModel(training_data, data_model, 1)
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
    (new_training_data['field_distances'], _) = core.buildRecordDistances(examples, fields)

    training_data = numpy.append(training_data, new_training_data)

    return training_data


def consoleLabel(uncertain_pairs, data_model):
    '''Command line interface for presenting and labeling training pairs by the user'''
    duplicates = []
    nonduplicates = []
    finished = False

    fields = [field for field in data_model['fields']
              if data_model['fields'][field]['type'] != 'Interaction']

    for record_pair in uncertain_pairs:
        label = ''

        for pair in record_pair:
            for field in fields:
                print field, ': ', pair[field]
            print ''

        print 'Do these records refer to the same thing?'

        valid_response = False
        while not valid_response:
            label = raw_input('(y)es / (n)o / (u)nsure / (f)inished\n')
            if label in ['y', 'n', 'u', 'f']:
                valid_response = True

        if label == 'y':
            duplicates.append(record_pair)
        elif label == 'n':
            nonduplicates.append(record_pair)
        elif label == 'f':
            print 'Finished labeling'
            finished = True
            break
        elif label != 'u':
            print 'Nonvalid response'
            raise

    return ({0: nonduplicates, 1: duplicates}, finished)


def semiSupervisedNonDuplicates(data_sample,
                                data_model,
                                nonduplicate_confidence_threshold=.7,
                                sample_size=2000):

    if len(data_sample) <= sample_size:
        return data_sample

    confident_distinct_pairs = []
    n_distinct_pairs = 0
    for pair in data_sample:

        pair_distance = core.recordDistances([pair], data_model)
        score = core.scorePairs(pair_distance, data_model)

        if score < 1 - nonduplicate_confidence_threshold:
            (key_pair, value_pair) = zip(*pair)
            confident_distinct_pairs.append(value_pair)
            n_distinct_pairs += 1
            if n_distinct_pairs == sample_size:
                return confident_distinct_pairs
