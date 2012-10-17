#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import lr
from affinegap import normalizedAffineGapDistance as stringDistance
import numpy
import json
from itertools import izip

def sampleDict(d, sample_size):

    if len(d) <= sample_size:
        return d

    sample_keys = random.sample(d.keys(), sample_size)
    return dict((k, d[k]) for k in d.keys() if k in sample_keys)

# using logistic regression, train weights for all fields in the data model

def trainModel(training_data, data_model, alpha=.001):

    (labels, fields, examples) = zip(*[(l, f, e) for (l, (f, e))
                                       in training_data])

    labels = numpy.array(labels, dtype='i4')
    examples = numpy.array(examples, dtype='f4')
    (weight, bias) = lr.lr(labels, examples, alpha)

    fields = sorted(data_model['fields'].keys())

    #weights = dict(zip(fields[0], weight))
    for i, name in enumerate(fields):
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model

#@profile
def recordDistances(candidates, data_model):

  # The record array has two elements, the first element is an array
  # of floats that has length equal the number of fields. The second
  # argument is an array of length 2 which stores the id of the
  # considered elements in the pair.

    fields = data_model['fields']

    field_dtype = [('names', 'a20', len(fields)), ('values', 'f4',
                   len(fields))]

    record_dtype = [('pairs', [('pair1', 'i4'), ('pair2', 'i4')]),
                    ('field_distances', field_dtype)]

    record_distances = numpy.zeros(len(candidates), dtype=record_dtype)
    

    key_pairs = []
    record_pairs = []


    [(key_pairs.append((candidate_1[0],
                        candidate_2[0])),
      record_pairs.append((candidate_1[1],
                           candidate_2[1])))
      for candidate_1, candidate_2 in candidates]


    record_distances = buildRecordDistances(record_pairs, fields, record_distances)
    record_distances['pairs'] = list(key_pairs)

    return record_distances

#@profile
def buildRecordDistances(record_pairs, fields, record_distances) :
  distances = numpy.zeros(1, dtype=record_distances['field_distances'].dtype)
  distances = distances['values'][0]



  field_distances = record_distances['field_distances']['values']
  sorted_fields = sorted(fields.keys())
  field_types = [fields[field]['type'] for field in sorted_fields]

  base_fields = []
  interactions = []
  if "Interaction" in field_types :  
    for i, name in enumerate(sorted_fields) :
      if fields[name]['type'] == "String" :
        base_fields.append(name)
      else :
        terms = fields[name]['interaction-terms']
        base_fields.append(terms[0])
        terms = [sorted_fields.index(term) for term in terms[1:]]
        interactions.append((i, terms))
  else :
    base_fields = sorted_fields

  if interactions :
    for (i, record_pair) in enumerate(record_pairs):
      record_1, record_2 = record_pair

      field_distances[i] = [stringDistance(record_1[name], record_2[name]) for name in base_fields]

      for j, term_indices in interactions :
        value = field_distances[i][j]
        for k in term_indices :
          value *= field_distances[i][k]
        field_distances[i][j] = value
  else :
    for (i, record_pair) in enumerate(record_pairs):
      record_1, record_2 = record_pair

      field_distances[i] = [stringDistance(record_1[name], record_2[name]) for name in base_fields]

  record_distances['field_distances']['values'] = field_distances
  return record_distances

def scorePairs(record_distances, data_model):
    fields = data_model['fields']
    field_names = sorted(data_model['fields'].keys())

    field_weights = [fields[name]['weight'] for name in field_names]
    bias = data_model['bias']

    field_distances = record_distances['field_distances']['values']

    scores = numpy.dot(field_distances, field_weights)

    scores = numpy.exp(scores + bias) / (1 + numpy.exp(scores + bias))

    return scores


# identify all pairs above a set threshold as duplicates

def scoreDuplicates(candidates,
                    data_model,
                    threshold=None,
                    ):

    record_distances = recordDistances(candidates, data_model)
    duplicate_scores = scorePairs(record_distances, data_model)

    pair_ids = [pair[0] for pair in record_distances]

    scored_pairs = zip(pair_ids, duplicate_scores)
    if threshold:
        return [pair for pair in scored_pairs if pair[1] > threshold]
    else:
        return scored_pairs


# define a data type for hashable dictionaries

class frozendict(dict):

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

    def __repr__(self):
        return 'frozendict(%s)' % dict.__repr__(self)
