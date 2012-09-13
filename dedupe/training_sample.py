#!/usr/bin/python
# -*- coding: utf-8 -*-
# provides functions for selecting a sample of training data

from random import sample, shuffle
from itertools import combinations
import blocking
import core
import numpy


# create a set of training data

def trainingDistances(training_pairs, data_model):
    fields = data_model['fields']

    field_dtype = [('names', 'a10', len(fields)), ('values', 'f4',
                   len(fields))]

    distances = numpy.zeros(1, dtype=field_dtype)

    training_data = []

    for (label, examples) in training_pairs.items():
        for (i, pair) in enumerate(examples):

            c_distances = core.calculateDistance(pair[0],
                                                 pair[1],
                                                 fields,
                                                 distances)
            c_distances = dict(zip(fields.keys(),
                                   c_distances[0]['values']))
            training_data.append((label, c_distances))

    return training_data


# create a random set of training pairs based on known duplicates

def randomTrainingPairs(data_d,
                        duplicates_s,
                        n_training_dupes,
                        n_training_distinct,
                        ):

    if n_training_dupes < len(duplicates_s):
        duplicates = sample(duplicates_s, n_training_dupes)
    else:
        duplicates = duplicates_s

    duplicates = [(data_d[tuple(pair)[0]], data_d[tuple(pair)[1]])
                  for pair in duplicates]

    all_pairs = list(combinations(data_d, 2))
    all_nonduplicates = set(all_pairs) - set(duplicates_s)

    nonduplicates = sample(all_nonduplicates, n_training_distinct)

    nonduplicates = [(data_d[pair[0]], data_d[pair[1]])
                     for pair in nonduplicates]

    return {0: nonduplicates, 1: duplicates}


# based on the data model and training we have so far, returns the n
# record pairs we are least certain about

def findUncertainPairs(record_distances, data_model):

    probability = core.scorePairs(record_distances, data_model)

    uncertainties = ((probability * numpy.log2(probability))
                     + ((1 - probability)
                        * numpy.log2(1 - probability))
                     )

    return numpy.argsort(uncertainties)


# loop for user to enter training data

def activeLearning(data_d,
                   data_model,
                   labelPairFunction,
                   num_questions,
                   training_data
                   ):
    duplicates = []
    nonduplicates = []
    pairs = blocking.allCandidates(data_d)
    record_distances = core.recordDistances(pairs, data_d, data_model)
    while finished == False :
        print 'finding the next uncertain pair ...'
        uncertain_indices = findUncertainPairs(record_distances,
                                               data_model)

    # pop the next most uncertain pair off of record distances

        record_distances = record_distances[:, uncertain_indices]
        uncertain_pair_ids = (record_distances['pairs'])[0:1]
        record_distances = record_distances[1:]

        uncertain_pairs = []
        for pair in uncertain_pair_ids :
            record_pair = [data_d[instance] for instance in pair]
            record_pair = [tuple(record_pair)]
            uncertain_pairs.append(record_pair)

        labeled_pairs, finished = labelPairFunction(uncertain_pairs,
                                          data_d,
                                          data_model)

        nonduplicates.extend(labeled_pairs[0])
        duplicates.extend(labeled_pairs[1])

        training_data = addTrainingData(labeled_pairs,
                                        data_model,
                                        training_data)

        data_model = core.trainModel(training_data, data_model)

    training_pairs = {0: nonduplicates, 1: duplicates}

    return (training_data, training_pairs, data_model)


# appends training data to the training data collection

def addTrainingData(labeled_pairs, data_model, training_data=[]):

    fields = data_model['fields']


    field_dtype = training_data.dtype[1]

    distances = numpy.zeros(1, dtype=field_dtype)

    num_training_pairs = len(labeled_pairs[0]) + len(labeled_pairs[1])

    new_training_data = numpy.zeros(num_training_pairs,
                                    dtype=training_data.dtype)

    i = 0
    for (label, examples) in labeled_pairs.items():
        for pair in examples:
            c_distances = core.calculateDistance(pair[0],
                                                 pair[1],
                                                 fields,
                                                 distances)

            example = (label, c_distances)
            new_training_data[i] = example
            i += 1

    training_data = numpy.append(training_data, new_training_data)

    return training_data


def consoleLabel(uncertain_pairs, data_model):
    duplicates = []
    nonduplicates = []
    finisehd = False

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
