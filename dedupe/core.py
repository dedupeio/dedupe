#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import lr
from affinegap import normalizedAffineGapDistance as stringDistance
import numpy
import json
import itertools

def sampleDict(d, sample_size):

    if len(d) <= sample_size:
        return d

    sample_keys = random.sample(d.keys(), sample_size)
    return dict((k, d[k]) for k in d.keys() if k in sample_keys)

# using logistic regression, train weights for all fields in the data model

def trainModel(training_data, data_model, alpha=.001):

    labels = training_data['label']
    examples = training_data['field_distances']['values']

    (weight, bias) = lr.lr(labels, examples, alpha)

    fields = sorted(data_model['fields'].keys())

    #weights = dict(zip(fields[0], weight))
    for i, name in enumerate(fields):
        data_model['fields'][name]['weight'] = float(weight[i])

    data_model['bias'] = bias

    return data_model

def recordDistances(candidates, data_model):

  # The record array has two elements, the first element is an array
  # of floats that has length equal the number of fields. The second
  # argument is an array of length 2 which stores the id of the
  # considered elements in the pair.



    fields = data_model['fields']

    field_dtype = [('names', 'a20', len(fields)), ('values', 'f4',
                   len(fields))]

    record_dtype = [('pairs', 'i4', 2),
                    ('field_distances', field_dtype)]


    candidates_1, candidates_2 = itertools.tee(candidates, 2)

    key_pairs = (candidate[0]
                 for candidate_pair in candidates_1
                 for candidate in candidate_pair)



    record_pairs = ((candidate_1[1], candidate_2[1])
                     for candidate_1, candidate_2
                     in candidates_2)




    field_distances, n_candidates = buildRecordDistances(record_pairs,
                                                         fields)
                                                         

    record_distances = numpy.zeros(n_candidates, dtype=record_dtype)

    record_distances['pairs'] = numpy.fromiter(key_pairs, 'i4').reshape(n_candidates, 2)
    record_distances['field_distances']['values'] = field_distances[0:n_candidates]

    return record_distances

def buildRecordDistances(record_pairs, fields) :
  n_fields = len(fields)

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


    field_distances = numpy.zeros((100000, n_fields)) 


    for (i, record_pair) in enumerate(record_pairs):
      if i % 100000 == 0 :
        field_distances = numpy.concatenate((field_distances,
                                               numpy.zeros((100000, n_fields))))
      record_1, record_2 = record_pair

      field_distances[i] = [stringDistance(record_1[name], record_2[name]) for name in base_fields]

      for j, term_indices in interactions :
        value = field_distances[i][j]
        for k in term_indices :
          value *= field_distances[i][k]
        field_distances[i][j] = value
  else :
    field_distances = numpy.fromiter((stringDistance(record_pair[0][name],
                                                     record_pair[1][name])
                                      for record_pair in record_pairs
                                      for name in base_fields),
                                     'f4')
    field_distances = field_distances.reshape(len(field_distances)/n_fields,
                                              n_fields)

  i = field_distances.shape[0] - 1
                                              

  return (field_distances, i+1)

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

    score_dtype = [('pairs', 'i4', 2), ('score', 'f4', 1)]
    scored_pairs = numpy.zeros(len(duplicate_scores),
                               dtype=score_dtype)

    scored_pairs['pairs'] = record_distances['pairs']
    scored_pairs['score'] = duplicate_scores

    scored_pairs = scored_pairs[scored_pairs['score'] > threshold]

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
