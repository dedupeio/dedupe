#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import warnings
import numpy
import collections

import dedupe.backport as backport
import dedupe.lr as lr

def randomPairsWithReplacement(n_records, sample_size) :
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("There may be duplicates in the sample")

    random_indices = numpy.random.randint(n_records, 
                                          size=sample_size*2)
    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return random_indices


def randomPairs(n_records, sample_size):
    """
    Return random combinations of indices for a square matrix of size
    n records
    """

    if n_records < 2 :
        raise ValueError("Needs at least two records")
    n = n_records * (n_records - 1) / 2

    if sample_size >= n:
        if sample_size > n :
            warnings.warn("Requested sample of size %d, only returning %d possible pairs" % (sample_size, n))

        random_indices = numpy.arange(n)
    elif 8 * n > numpy.iinfo('uint').max :
        return randomPairsWithReplacement(n_records, sample_size)
    else :
        try:
            random_indices = numpy.random.randint(n, size=sample_size)
            random_indices.dtype = 'uint'
        except OverflowError:
            return randomPairsWithReplacement(n_records, sample_size)


    b = 1 - 2 * n_records

    x = numpy.trunc((-b - numpy.sqrt(b ** 2 - 8 * random_indices)) / 2)
    y = random_indices + x * (b + x + 2) / 2 + 1

    return numpy.column_stack((x, y)).astype(int)

def randomPairsMatch(n_records_A, n_records_B, sample_size):
    """
    Return random combinations of indices for record list A and B
    """
    if not n_records_A or not n_records_B :
        raise ValueError("There must be at least one record in A and in B")

    if sample_size >= n_records_A * n_records_B :

        if sample_size > n_records_A * n_records_B :
            warnings.warn("Requested sample of size %d, only returning %d possible pairs" % (sample_size, n_records_A * n_records_B))

        return backport.cartesian((numpy.arange(n_records_A),
                                   numpy.arange(n_records_B)))

    A_samples = numpy.random.randint(n_records_A, size=sample_size)
    B_samples = numpy.random.randint(n_records_B, size=sample_size)
    pairs = zip(A_samples,B_samples)
    set_pairs = set(pairs)

    while len(set_pairs) < sample_size:
        set_pairs.update(randomPairsMatch(n_records_A,
                                          n_records_B,
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


def fieldDistances(record_pairs, data_model, num_records=None):

    if num_records is None :
        num_records = len(record_pairs)
    
    distances = numpy.empty((num_records, data_model.total_fields))
    
    current_column = 0

    field_comparators = data_model.field_comparators
    for i, (record_1, record_2) in enumerate(record_pairs) :
        for j, (field, compare) in enumerate(field_comparators) :
            distances[i,j] = compare(record_1[field],
                                     record_2[field])

    current_column += len(field_comparators)

    for cat_index, length in data_model.categorical_indices :
        start = current_column
        end = start + (length - 2)
        
        distances[:,start:end] =\
                (distances[:, cat_index][:,None] 
                 == numpy.arange(2, length)[None,:])

        distances[:,cat_index][distances[:,cat_index] > 1] = 0
                             
        current_column = end


    for interaction in data_model.interactions :
        distances[:,current_column] =\
                numpy.prod(distances[:,interaction], axis=1)

        current_column += 1

    missing_data = numpy.isnan(distances[:,:current_column])

    distances[:,:current_column][missing_data] = 0

    distances[:,current_column:] =\
            1 - missing_data[:,data_model.missing_field_indices]

    return distances

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

    def __call__(self, chunk_queue, scored_pairs_queue) :
        while True :
            record_pairs = chunk_queue.get()
            if record_pairs is None :
                # put the poison bill back in the queue so that other
                # scorers will know to stop
                chunk_queue.put(None)
                break
            scored_pairs = self.scoreRecords(record_pairs)
            scored_pairs_queue.put(scored_pairs)

    def scoreRecords(self, record_pairs) :
        num_records = len(record_pairs)

        scored_pairs = numpy.empty(num_records,
                                   dtype = self.dtype)

        def split_records() :
            for i, pair in enumerate(record_pairs) :
                record_1, record_2 = pair
                scored_pairs['pairs'][i] = (record_1[0], record_2[0])
                yield (record_1[1], record_2[1])

        scored_pairs['score'] = scorePairs(fieldDistances(split_records(), 
                                                          self.data_model,
                                                          num_records),
                                           self.data_model)

        scored_pairs = scored_pairs[scored_pairs['score'] > self.threshold]   

        return scored_pairs

def scoreDuplicates(records, data_model, num_processes, threshold=0):
    records = iter(records)

    chunk_size = 1000

    queue_size = num_processes
    record_pairs_queue = backport.Queue(queue_size)
    scored_pairs_queue = backport.Queue()

    id_type, records = idType(records)
    
    score_dtype = [('pairs', id_type, 2), ('score', 'f4', 1)]

    scoring_function = ScoringFunction(data_model, 
                                       threshold,
                                       score_dtype)

    # Start processes
    processes = [backport.Process(target=scoring_function, 
                                  args=(record_pairs_queue, 
                                        scored_pairs_queue))
                 for _ in xrange(num_processes)]

    [process.start() for process in processes]

    num_chunks = 0
    num_records = 0

    while True :
        chunk = list(itertools.islice(records, chunk_size))
        if chunk :
            record_pairs_queue.put(chunk)
            num_chunks += 1
            num_records += chunk_size

            if num_chunks > queue_size :
                if record_pairs_queue.full() :
                    if chunk_size < 100000 :
                        if num_chunks % 10 == 0 :
                            chunk_size = int(chunk_size * 1.1)
                else :
                    if chunk_size > 100 :
                        chunk_size = int(chunk_size * 0.9)
        else :
            # put poison pill in queue to tell scorers that they are
            # done
            record_pairs_queue.put(None)
            break

    scored_pairs = numpy.empty(num_records,
                               dtype=score_dtype)

    start = 0
    for _ in xrange(num_chunks) :
        score_chunk = scored_pairs_queue.get()
        end = start + len(score_chunk)
        scored_pairs[start:end,] = score_chunk
        start = end

    scored_pairs.resize((end,))

    [process.join() for process in processes]

    return scored_pairs


def idType(records) :
    record, records = peek(records)

    id_type = type(record[0][0])
    if id_type is str or id_type is unicode :
        id_type = (unicode, len(record[0][0]) + 5)

    return numpy.dtype(id_type), records


def peek(records) :
    try :
        record = records.next()
    except AttributeError as e:
        if "has no attribute 'next'" not in str(e) :
            raise
        try :
            records = iter(records)
            record = records.next()
        except StopIteration :
            return None, records


    return record, itertools.chain([record], records)


def freezeData(data) : # pragma : no cover
    return [(frozendict(record_1), 
             frozendict(record_2))
            for record_1, record_2 in data]



class frozendict(collections.Mapping):
    """Don't forget the docstrings!!"""

    def __init__(self, *args, **kwargs): # pragma : no cover
        self._d = dict(*args, **kwargs)

    def __iter__(self):                  # pragma : no cover
        return iter(self._d)

    def __len__(self):                   # pragma : no cover
        return len(self._d)

    def __getitem__(self, key):          # pragma : no cover
        return self._d[key]

    def __repr__(self) :
        return '<frozendict %s>' % repr(self._d)

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(frozenset(self._d.iteritems()))
            return h
