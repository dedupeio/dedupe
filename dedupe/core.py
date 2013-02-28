#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import random
import json
import itertools
import logging

import numpy

import lr
from affinegap import normalizedAffineGapDistance as stringDistance


def randomPairs(n_records, sample_size, zero_indexed=True):
    '''Return random combinations of indicies for a square matrix of size n records'''
    n = n_records * (n_records - 1) / 2

    if sample_size >= n:
        random_indices = numpy.arange(n)
        numpy.random.shuffle(random_indices)
    else:
        random_indices = numpy.array(random.sample(xrange(n), sample_size))

    b = 1 - 2 * n_records

    x = numpy.trunc((-b - numpy.sqrt(b ** 2 - 8 * random_indices)) / 2)
    y = random_indices + x * (b + x + 2) / 2 + 1

    if not zero_indexed:
        x += 1
        y += 1

    return numpy.column_stack((x, y))


def dataSample(d, sample_size):

    random_pairs = randomPairs(len(d), sample_size)

    return tuple(((k_1, d[k_1]), (k_2, d[k_2])) for (k_1, k_2) in random_pairs)


def trainModel(training_data, data_model, alpha=.001):
    '''Use logistic regression to train weights for all fields in the data model'''
    labels = training_data['label']
    examples = training_data['distances']

    (weight, bias) = lr.lr(labels, examples, alpha)

    fields = sorted(data_model['fields'].keys())

    for (i, name) in enumerate(fields):
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model


def fieldDistances(candidates, data_model):
    fields = data_model['fields']
    record_dtype = [('pairs', 'i4', 2), ('distances', 'f4', (len(fields), ))]

    (candidates_1, candidates_2) = itertools.tee(candidates, 2)

    key_pairs = (candidate[0] for candidate_pair in candidates_1
                 for candidate in candidate_pair)

    record_pairs = ((candidate_1[1], candidate_2[1]) 
                    for (candidate_1, candidate_2) in candidates_2)

    _field_distances = buildFieldDistances(record_pairs, fields)

    field_distances = numpy.zeros(_field_distances.shape[0],
                                  dtype=record_dtype)

    field_distances['pairs'] = numpy.fromiter(key_pairs, 'i4').reshape(-1, 2)
    field_distances['distances'] = _field_distances

    return field_distances


def buildFieldDistances(record_pairs, fields):

    field_comparators = [(field, v['comparator'])
                         for field, v in fields.items()]

    
    field_distances = numpy.fromiter((compare(record_pair[0][field],
                                              record_pair[1][field]) 
                                      for record_pair in record_pairs 
                                      for field, compare in field_comparators), 
                                     'f4')
    field_distances = field_distances.reshape(-1,len(fields))

    return field_distances

def scorePairs(field_distances, data_model):
    fields = data_model['fields']
    field_names = sorted(data_model['fields'].keys())

    field_weights = [fields[name]['weight'] for name in field_names]
    bias = data_model['bias']

    distances = field_distances['distances']

    scores = numpy.dot(distances, field_weights)

    scores = numpy.exp(scores + bias) / (1 + numpy.exp(scores + bias))

    return scores


def scoreDuplicates(candidates, data_model, threshold=None):

    score_dtype = [('pairs', 'i4', 2), ('score', 'f4', 1)]
    scored_pairs = numpy.zeros(0, dtype=score_dtype)

    complete = False
    chunk_size = 5000
    i = 1
    while not complete:

        can_slice = list(itertools.islice(candidates, 0, chunk_size))

        field_distances = fieldDistances(can_slice, data_model)
        duplicate_scores = scorePairs(field_distances, data_model)

        scored_pairs = numpy.append(scored_pairs,
                                    numpy.array(zip(field_distances['pairs'],
                                                    duplicate_scores),
                                                dtype=score_dtype)[duplicate_scores > threshold], 
                                    axis=0)
        i += 1
        if len(field_distances) < chunk_size:
            complete = True
            logging.info('num chunks %d' % i)

    logging.info('all scores %d' % scored_pairs.shape)
    scored_pairs = numpy.unique(scored_pairs)
    logging.info('unique scores %d' % scored_pairs.shape)

    return scored_pairs


class frozendict(dict):
    '''
    A data type for hashable dictionaries
    From http://code.activestate.com/recipes/414283-frozen-dictionaries/
    '''

    def _blocked_attribute(obj):
        raise AttributeError('A frozendict cannot be modified.')

    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args):
        new = dict.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args):
        pass

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(tuple(sorted(self.items())))
            return h
