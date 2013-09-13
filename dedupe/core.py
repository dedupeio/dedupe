#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import random
import json
import itertools
import logging
from itertools import count
import warnings

import numpy

import lr
from dedupe.distance.affinegap import normalizedAffineGapDistance as stringDistance

def randomPairs(n_records, sample_size, zero_indexed=True):
    """
    Return random combinations of indices for a square matrix of size
    n records
    """

    if n_records < 2 :
        raise ValueError("Needs at least two records")
    n = n_records * (n_records - 1) / 2

    if sample_size >= n:
        warnings.warn("Requested sample of size %d, only returning %d possible pairs" % (sample_size, n))

        random_indices = numpy.arange(n)
        numpy.random.shuffle(random_indices)
    else:
        try:
            random_indices = numpy.array(random.sample(xrange(n), sample_size))
        except OverflowError:
            # If the population is very large relative to the sample
            # size than we'll get very few duplicates by chance
            logging.warning("There may be duplicates in the sample")
            sample = numpy.array([random.sample(xrange(n_records), 2)
                                  for _ in xrange(sample_size)])
            return numpy.sort(sample, axis=1)



    b = 1 - 2 * n_records

    x = numpy.trunc((-b - numpy.sqrt(b ** 2 - 8 * random_indices)) / 2)
    y = random_indices + x * (b + x + 2) / 2 + 1

    if not zero_indexed:
        x += 1
        y += 1

    return numpy.column_stack((x, y)).astype(int)



def trainModel(training_data, data_model, alpha=.001):
    """
    Use logistic regression to train weights for all fields in the data model
    """
    
    labels = training_data['label']
    examples = training_data['distances']

    (weight, bias) = lr.lr(labels, examples, alpha)

    for i, name in enumerate(data_model['fields']) :
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model


def fieldDistances(record_pairs, data_model):
    fields = data_model['fields']

    field_comparators = [(field, v['comparator'])
                         for field, v in fields.items()
                         if v['type'] not in ('Missing Data',
                                              'Interaction')]

    
    missing_field_indices = [i for i, (field, v) 
                             in enumerate(fields.items())
                             if 'Has Missing' in v and v['Has Missing']]

    field_names = fields.keys()
  
    interactions = []
    for field in fields :
        if fields[field]['type'] == 'Interaction' :
            interaction_indices = []
            for interaction_field in fields[field]['Interaction Fields'] :
                interaction_indices.append(field_names.index(interaction_field))
            interactions.append(interaction_indices)
    
    field_distances = numpy.fromiter((compare(record_pair[0][field],
                                              record_pair[1][field]) 
                                      for record_pair in record_pairs 
                                      for field, compare in field_comparators), 
                                     'f4')
    field_distances = field_distances.reshape(-1,len(field_comparators))

    interaction_distances = numpy.empty((field_distances.shape[0],
                                         len(interactions)))

    for i, interaction in enumerate(interactions) :
        a = numpy.prod(field_distances[...,interaction], axis=1)
        interaction_distances[...,i] = a
       
    field_distances = numpy.concatenate((field_distances,
                                         interaction_distances),
                                        axis=1)

        

    missing_data = numpy.isnan(field_distances)

    field_distances[missing_data] = 0

    missing_indicators = 1-missing_data[:,missing_field_indices]

    

    field_distances = numpy.concatenate((field_distances,
                                         1-missing_data[:,missing_field_indices]),
                                        axis=1)

    return field_distances


def scorePairs(field_distances, data_model):
    fields = data_model['fields']

    field_weights = [fields[name]['weight'] for name in fields]
    bias = data_model['bias']

    scores = numpy.dot(field_distances, field_weights)

    scores = numpy.exp(scores + bias) / (1 + numpy.exp(scores + bias))

    return scores


def scoreDuplicates(ids, records, id_type, data_model, threshold=None):

    score_dtype = [('pairs', id_type, 2), ('score', 'f4', 1)]
    scored_pairs = numpy.zeros(0, dtype=score_dtype)

    complete = False
    chunk_size = 5000
    i = 1
    while not complete:
        id_slice = list(itertools.islice(ids, 0, chunk_size))
        can_slice = list(itertools.islice(records, 0, chunk_size))

        field_distances = fieldDistances(can_slice, data_model)
        duplicate_scores = scorePairs(field_distances, data_model)

        scored_pairs = numpy.append(scored_pairs,
                                    numpy.array(zip(id_slice,
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

def blockedPairs(blocks) :
    for block in blocks :

        block_pairs = itertools.combinations(block, 2)

        for pair in block_pairs :
            yield pair

def split(iterable):
    it = iter(iterable)
    q = [collections.deque([x]) for x in it.next()] 
    def proj(qi):
        while True:
            if not qi:
                for qj, xj in zip(q, it.next()):
                    qj.append(xj)
            yield qi.popleft()
    for qi in q:
        yield proj(qi)



import collections

class frozendict(collections.Mapping):
    """Don't forget the docstrings!!"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __repr__(self) :
        return '<frozendict %s>' % repr(self._d)

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(frozenset(self._d.iteritems()))
            return h




        
