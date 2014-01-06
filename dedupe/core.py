#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import json
import itertools
import logging
import warnings
import multiprocessing
import Queue
import numpy
import time
import collections

import lr

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

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

def randomPairsMatch(n_records_A, n_records_B, sample_size):
    """
    Return random combinations of indices for record list A and B
    """

    A_samples = numpy.random.randint(n_records_A, size=sample_size)
    B_samples = numpy.random.randint(n_records_B, size=sample_size)
    pairs = zip(A_samples,B_samples)
    set_pairs = set(pairs)

    if len(set_pairs) < sample_size:
        return set.union(set_pairs,
                         randomPairsMatch(n_records_A,n_records_B,
                                          (sample_size-len(set_pairs))))
    else:
        return set_pairs


def trainModel(training_data, data_model, alpha=.001):
    """
    Use logistic regression to train weights for all fields in the data model
    """
    
    labels = numpy.array(training_data['label'] == 'match', dtype='i4')
    examples = training_data['distances']

    (weight, bias) = lr.lr(labels, examples, alpha)

    for i, name in enumerate(data_model['fields']) :
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model


def fieldDistances(record_pairs, data_model):
    # Think about filling this in instead of concatenating

    field_distances = numpy.fromiter((compare(record_pair[0][field],
                                              record_pair[1][field]) 
                                      for record_pair in record_pairs 
                                      for field, compare in data_model.field_comparators
                                      if record_pair), 
                                     'f4')

    field_distances = field_distances.reshape(-1,
                                              len(data_model.field_comparators))

    for cat_index, length in data_model.categorical_indices :
        different_sources = field_distances[:, cat_index][...,None] == numpy.arange(2, length)[None,...]


        field_distances[:, cat_index][field_distances[:, cat_index] > 1] = 0
        field_distances = numpy.concatenate((field_distances,
                                             different_sources.astype(float)),
                                            axis=1)


    interaction_distances = numpy.empty((field_distances.shape[0],
                                         len(data_model.interactions)))

    for i, interaction in enumerate(data_model.interactions) :
        a = numpy.prod(field_distances[...,interaction], axis=1)
        interaction_distances[...,i] = a
       
    field_distances = numpy.concatenate((field_distances,
                                         interaction_distances),
                                        axis=1)

        

    missing_data = numpy.isnan(field_distances)

    field_distances[missing_data] = 0

    missing_indicators = 1-missing_data[:,data_model.missing_field_indices]
    

    field_distances = numpy.concatenate((field_distances,
                                         missing_indicators),
                                        axis=1)

    return field_distances


def scorePairs(field_distances, data_model):
    fields = data_model['fields']

    field_weights = [fields[name]['weight'] for name in fields]
    bias = data_model['bias']

    scores = numpy.dot(field_distances, field_weights)

    scores = numpy.exp(scores + bias) / (1 + numpy.exp(scores + bias))

    return scores

class ScoringFunction(object) :
    def __init__(self, data_model, threshold, dtype) :
        self.data_model = data_model
        self.threshold = threshold
        self.dtype = dtype

    def __call__(self, record_pairs) :
        ids = []

        def split_records() :
            for pair in record_pairs :
                if pair :
                    ids.append((pair[0][0], pair[1][0]))
                    yield (pair[0][1], pair[1][1])

        scores = scorePairs(fieldDistances(split_records(), 
                                           self.data_model),
                            self.data_model)

        filtered_scores = ((pair_id, score) 
                           for pair_id, score in itertools.izip(ids, scores) 
                           if score > self.threshold)

        filtered_scores = list(filtered_scores)

        scored_pairs = numpy.fromiter(filtered_scores,
                                      dtype=self.dtype)

        return scored_pairs

def scoreDuplicates(records, data_model, pool, threshold=0):

    record, records = peek(records)

    id_type = idType(record)
    
    score_dtype = [('pairs', id_type, 2), ('score', 'f4', 1)]

    scored_pairs = numpy.empty((0,), dtype=score_dtype)

    record_chunks = grouper(records, 1000000)

    scoring_function = ScoringFunction(data_model, 
                                       threshold,
                                       score_dtype)

    score_queue = multiprocessing.Queue()

    results = [pool.apply_async(scoring_function,
                               (chunk,),
                               callback=score_queue.put)
               for chunk in record_chunks] 

    while not all([r.ready() for r in results]) :
        try :
            # equivalent to numpy.union1d(a,b)
            # http://stackoverflow.com/questions/12427146/combine-two-arrays-and-sort
            scored_pairs = numpy.concatenate((scored_pairs, 
                                              score_queue.get(True, 1)))
            scored_pairs.sort()
            flag = numpy.ones(len(scored_pairs), dtype=bool)
            numpy.not_equal(scored_pairs[1:], 
                            scored_pairs[:-1], 
                            out=flag[1:])
            scored_pairs[flag]

        except Queue.Empty :
            pass

    score_queue.close()
            
    return scored_pairs

def idType(record) :
    id_type = type(record[0][0])
    if id_type is str :
        id_type = (str, len(record[0][0]) + 5)

    return numpy.dtype(id_type)


def peek(records) :
    try :
        record = records.next()
    except AttributeError as e:
        if "has no attribute 'next'" not in str(e) :
            raise
        records = iter(records)
        record = records.next()

    return record, itertools.chain([record], records)



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




# class frozendict(dict):
#     def _blocked_attribute(obj):
#         raise AttributeError, "A frozendict cannot be modified."

#     def __new__(cls, *args):
#         new = dict.__new__(cls)
#         dict.__init__(new, *args)
#         return new

#     def __init__(self, *args):
#         pass

#     def __hash__(self):
#         try:
#             return self._cached_hash
#         except AttributeError:
#             h = self._cached_hash = hash(tuple(sorted(self.items())))
#             return h

#     def __repr__(self):
#         return "frozendict(%s)" % dict.__repr__(self)

