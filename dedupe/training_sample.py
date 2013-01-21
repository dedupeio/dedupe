#!/usr/bin/python
# -*- coding: utf-8 -*-
# provides functions for selecting a sample of training data

from random import sample, shuffle
from itertools import combinations
import blocking
import core
import numpy

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
                   training_data,
                   training_pairs = None,
                   key_groups = []
                   ):

    duplicates = []
    nonduplicates = []

    if training_pairs :
        nonduplicates.extend(training_pairs[0])
        duplicates.extend(training_pairs[1])

    finished = False
    candidates = blocking.allCandidates(data_d, key_groups)

    import time
    t_train = time.time()
    record_distances = core.recordDistances(candidates, data_model)
    print 'calculated recordDistances in ', time.time() - t_train, 'seconds'
    
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
            record_pair = tuple(record_pair)
            uncertain_pairs.append(record_pair)

        labeled_pairs, finished = labelPairFunction(uncertain_pairs,
                                          data_model)

        nonduplicates.extend(labeled_pairs[0])
        duplicates.extend(labeled_pairs[1])

        training_data = addTrainingData(labeled_pairs,
                                        data_model,
                                        training_data)
        if len(training_data) > 0 :
            data_model = core.trainModel(training_data, data_model)
        else :
            raise ValueError("No training pairs given")

    training_pairs = {0: nonduplicates, 1: duplicates}

    return (training_data, training_pairs, data_model)


# appends training data to the training data collection

def addTrainingData(labeled_pairs, data_model, training_data=[]):

    fields = data_model['fields']

    n_distinct_pairs, n_dupe_pairs = len(labeled_pairs[0]), len(labeled_pairs[1])

    new_training_data = numpy.zeros(n_distinct_pairs + n_dupe_pairs,
                                    dtype=training_data.dtype)

    labels = labeled_pairs.keys()
    examples = [record_pair for example in labeled_pairs.values() for record_pair in example]

    new_training_data['label'] = [labels[0]] * n_distinct_pairs + [labels[1]] * n_dupe_pairs
    new_training_data['field_distances']['values'] = core.buildRecordDistances(examples, fields)[0] 

    training_data = numpy.append(training_data, new_training_data)

    return training_data


def consoleLabel(uncertain_pairs, data_model):
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
