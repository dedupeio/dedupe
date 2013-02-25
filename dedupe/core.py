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
    examples = training_data['field_distances']

    (weight, bias) = lr.lr(labels, examples, alpha)

    fields = sorted(data_model['fields'].keys())

    for (i, name) in enumerate(fields):
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model


def recordDistances(candidates, data_model):
    fields = data_model['fields']
    record_dtype = [('pairs', 'i4', 2), ('field_distances', 'f4', (len(fields), ))]

    (candidates_1, candidates_2) = itertools.tee(candidates, 2)

    key_pairs = (candidate[0] for candidate_pair in candidates_1
                 for candidate in candidate_pair)

    record_pairs = ((candidate_1[1], candidate_2[1]) 
                    for (candidate_1, candidate_2) in candidates_2)

    (field_distances, n_candidates) = buildRecordDistances(record_pairs, fields)

    record_distances = numpy.zeros(n_candidates, dtype=record_dtype)

    record_distances['pairs'] = numpy.fromiter(key_pairs, 'i4').reshape(-1, 2)
    record_distances['field_distances'] = field_distances[0:n_candidates]

    return record_distances


def buildRecordDistances(record_pairs, fields):
    n_fields = len(fields)

    sorted_fields = sorted(fields.keys())
    field_types = [fields[field]['type'] for field in sorted_fields]

    base_fields = []
    interactions = []
    if 'Interaction' in field_types:
        for (i, name) in enumerate(sorted_fields):
            if fields[name]['type'] == 'String':
                base_fields.append(name)
            else:
                terms = fields[name]['interaction-terms']
                base_fields.append(terms[0])
                terms = [sorted_fields.index(term) for term in terms[1:]]
                interactions.append((i, terms))
    else:
        base_fields = sorted_fields

    if interactions:
        field_distances = numpy.zeros((100000, n_fields))

        for (i, record_pair) in enumerate(record_pairs):
            if i % 100000 == 0:
                field_distances = numpy.concatenate((field_distances, 
                                                     numpy.zeros((100000, n_fields))))
            (record_1, record_2) = record_pair

            field_distances[i] = [stringDistance(record_1[name], record_2[name]) 
                                  for name in base_fields]

            for (j, term_indices) in interactions:
                value = field_distances[i][j]
                for k in term_indices:
                    value *= field_distances[i][k]
                field_distances[i][j] = value
    else:
        field_distances = numpy.fromiter((stringDistance(record_pair[0][name], record_pair[1][name]) 
                                          for record_pair in record_pairs 
                                          for name in base_fields), 
                                         'f4')
        field_distances = field_distances.reshape(-1, n_fields)

    i = field_distances.shape[0] - 1

    return (field_distances, i + 1)


def scorePairs(record_distances, data_model):
    fields = data_model['fields']
    field_names = sorted(data_model['fields'].keys())

    field_weights = [fields[name]['weight'] for name in field_names]
    bias = data_model['bias']

    field_distances = record_distances['field_distances']

    scores = numpy.dot(field_distances, field_weights)

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

        record_distances = recordDistances(can_slice, data_model)
        duplicate_scores = scorePairs(record_distances, data_model)

        scored_pairs = numpy.append(scored_pairs,
                                    numpy.array(zip(record_distances['pairs'],
                                                    duplicate_scores),
                                                dtype=score_dtype)[duplicate_scores > threshold], 
                                    axis=0)
        i += 1
        if len(record_distances) < chunk_size:
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
