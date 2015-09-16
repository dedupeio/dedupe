#!/usr/bin/python
# -*- coding: utf-8 -*-
from builtins import range, next, zip
from future.utils import viewvalues
import sys
if sys.version < '3':
    text_type = unicode
    binary_type = str
else:
    text_type = str
    binary_type = bytes
    unicode = str

import itertools
import warnings
import numpy
import collections
import time
import tempfile
import os

import dedupe.backport as backport

class ChildProcessError(Exception) :
    pass

def randomPairsWithReplacement(n_records, sample_size) :
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("There may be duplicates in the sample")

    try :
        random_indices = numpy.random.randint(n_records, 
                                              size=sample_size*2)
    except OverflowError:
        max_int = numpy.iinfo('int').max
        warnings.warn("Asked to sample pairs from %d records, will only sample pairs from first %d records" % (n_records, max_int))
        random_indices = numpy.random.randint(max_int, 
                                              size=sample_size*2)


        
    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return [(p.item(), q.item()) for p, q in random_indices]


def randomPairs(n_records, sample_size):
    """
    Return random combinations of indices for a square matrix of size
    n records
    """

    if n_records < 2 :
        raise ValueError("Needs at least two records")
    n = n_records * (n_records - 1) / 2

    # numpy doesn't always throw an overflow error so we need to 
    # check to make sure that the largest number we'll use is smaller
    # than the numpy's maximum unsigned integer
    if 8 * n > numpy.iinfo('uint').max :
        return randomPairsWithReplacement(n_records, sample_size)

    if sample_size >= n:
        if sample_size > n :
            warnings.warn("Requested sample of size %d, only returning %d possible pairs" % (sample_size, n))

        random_indices = numpy.arange(n)
    else :
        random_indices = numpy.random.randint(int(n), size=sample_size)

    random_indices = random_indices.astype('uint')

    b = 1 - 2 * n_records

    x = numpy.trunc((-b - numpy.sqrt(b ** 2 - 8 * random_indices)) / 2)
    y = random_indices + x * (b + x + 2) / 2 + 1

    stacked = numpy.column_stack((x, y)).astype(int)

    return [(p.item(), q.item()) for p, q in stacked]

def randomPairsMatch(n_records_A, n_records_B, sample_size):
    """
    Return random combinations of indices for record list A and B
    """
    if not n_records_A or not n_records_B :
        raise ValueError("There must be at least one record in A and in B")

    if sample_size >= n_records_A * n_records_B :

        if sample_size > n_records_A * n_records_B :
            warnings.warn("Requested sample of size %d, only returning %d possible pairs" % (sample_size, n_records_A * n_records_B))

        return cartesian((numpy.arange(n_records_A),
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


def trainModel(training_data, data_model, learner=None, alpha=.001):
    """
    Use logistic regression to train weights for all fields in the data model
    """
    
    labels = numpy.array(training_data['label'] == b'match', dtype='i4')
    examples = training_data['distances']

    weight, bias = learner(labels, examples, alpha)

    for i, field_definition in enumerate(data_model['fields']) :
        field_definition.weight = float(weight[i])

    data_model['bias'] = bias

    return data_model

def fieldDistances(record_pairs, data_model=None):
    num_records = len(record_pairs)

    distances = numpy.empty((num_records, data_model.n_fields))
    field_comparators = data_model.field_comparators

    for i, (record_1, record_2) in enumerate(record_pairs) :
        
        for field, compare, start, stop in field_comparators :
            if record_1[field] is not None and record_2[field] is not None :
                distances[i,start:stop] = compare(record_1[field],
                                                  record_2[field])
            elif hasattr(compare, 'missing') :
                distances[i,start:stop] = compare(record_1[field],
                                                  record_2[field])
            else :
                distances[i,start:stop] = numpy.nan

    
    distances = derivedDistances(distances, data_model)

    return distances

def derivedDistances(primary_distances, data_model) :
    distances = primary_distances

    current_column = data_model.derived_start

    for interaction in data_model.interactions :
        distances[:,current_column] =\
                numpy.prod(distances[:,interaction], axis=1)

        current_column += 1

    missing_data = numpy.isnan(distances[:,:current_column])

    distances[:,:current_column][missing_data] = 0

    if data_model.missing_field_indices :
        distances[:,current_column:] =\
            1 - missing_data[:,data_model.missing_field_indices]

    return distances


def scorePairs(field_distances, data_model):
    fields = data_model['fields']

    field_weights = [field.weight for field in fields]
    bias = data_model['bias']

    scores = numpy.dot(field_distances, field_weights)

    scores = numpy.exp(scores + bias) / (1 + numpy.exp(scores + bias))

    return scores

class ScoreRecords(object) :
    def __init__(self, data_model, threshold) :
        self.data_model = data_model
        self.threshold = threshold
        self.score_queue = None

    def __call__(self, records_queue, score_queue) :
        self.score_queue = score_queue
        while True :
            record_pairs = records_queue.get()
            if record_pairs is None :
                break

            try :
                filtered_pairs = self.fieldDistance(record_pairs)
                if filtered_pairs is not None :
                    score_queue.put(filtered_pairs)
            except Exception as e :
                score_queue.put(e)
                raise

        score_queue.put(None)

    def fieldDistance(self, record_pairs) :
        ids = []
        records = []
        
        for record_pair in record_pairs :
            ((id_1, record_1, smaller_ids_1), 
             (id_2, record_2, smaller_ids_2)) = record_pair

            if set.isdisjoint(smaller_ids_1, smaller_ids_2) :
                
                ids.append((id_1, id_2))
                records.append((record_1, record_2))

        if records :
            distances = fieldDistances(records, self.data_model)
            scores = scorePairs(distances, self.data_model)

            scored_pairs = numpy.rec.fromarrays((ids, scores),
                                                dtype= [('pairs', 'object', 2), 
                                                        ('score', 'f4', 1)])
            
            filtered_pairs = scored_pairs[scores > self.threshold]

            return filtered_pairs

def mergeScores(score_queue, result_queue, stop_signals) :
    scored_pairs = numpy.empty(0, dtype= [('pairs', 'object', 2), 
                                          ('score', 'f4', 1)])

    seen_signals = 0
    while seen_signals < stop_signals  :
        score_chunk = score_queue.get()
        if isinstance(score_chunk, Exception) :
            result_queue.put(score_chunk)
            return

        if score_chunk is not None :
            scored_pairs = numpy.concatenate((scored_pairs, score_chunk))
        else :
            seen_signals += 1

    if len(scored_pairs) :
        python_type = type(scored_pairs['pairs'][0][0])
        if python_type is binary_type or python_type is text_type :
            max_length = len(max(numpy.ravel(scored_pairs['pairs']), key=len))
            python_type = (unicode, max_length)
        
        write_dtype = [('pairs', python_type, 2),
                       ('score', 'f4', 1)]

        scored_pairs = scored_pairs.astype(write_dtype)

        scored_pairs_file, file_path = tempfile.mkstemp()
        
        os.close(scored_pairs_file)

        fp = numpy.memmap(file_path, 
                          dtype=scored_pairs.dtype, 
                          shape=scored_pairs.shape)
        fp[:] = scored_pairs[:]

        result_queue.put((file_path, scored_pairs.dtype))

    else :
        result_queue.put(scored_pairs)

def scoreDuplicates(records, data_model, num_cores=1, threshold=0) :
    if num_cores < 2 :
        from multiprocessing.dummy import Process, Pool, Queue
        SimpleQueue = Queue
    else :
        from .backport import Process, Pool, SimpleQueue

    record_pairs_queue = SimpleQueue()
    score_queue =  SimpleQueue()
    result_queue = SimpleQueue()

    n_map_processes = max(num_cores-1, 1)
    score_records = ScoreRecords(data_model, threshold) 
    map_processes = [Process(target=score_records,
                             args=(record_pairs_queue,
                                   score_queue))
                     for _ in range(n_map_processes)]
    [process.start() for process in map_processes]

    reduce_process = Process(target=mergeScores,
                             args=(score_queue,
                                   result_queue,
                                   n_map_processes))
    reduce_process.start()

    fillQueue(record_pairs_queue, records, n_map_processes)

    result = result_queue.get()
    if isinstance(result, Exception) :
        raise ChildProcessError

    if result :
        scored_pairs_file, dtype = result
        scored_pairs = numpy.memmap(scored_pairs_file,
                                    dtype=dtype)
    else :
        scored_pairs = result

    return scored_pairs



def fillQueue(queue, iterable, stop_signals) :
    iterable = iter(iterable)
    chunk_size = 100000
    multiplier = 1.1

    # initial values
    i = 0
    n_records = 0
    t0 = time.clock()
    last_rate = 10000

    while True :
        chunk = list(itertools.islice(iterable, int(chunk_size)))
        if chunk :
            queue.put(chunk)

            n_records += chunk_size
            i += 1

            if i % 10 :
                time_delta = max(time.clock() - t0, 0.0001)

                current_rate = n_records/time_delta

                # chunk_size is always either growing or shrinking, if
                # the shrinking led to a faster rate, keep
                # shrinking. Same with growing. If the rate decreased,
                # reverse directions
                if current_rate < last_rate :
                    multiplier = 1/multiplier

                chunk_size = max(chunk_size * multiplier, 1)

                last_rate = current_rate
                n_records = 0
                t0 = time.clock()
                

        else :
            # put poison pills in queue to tell scorers that they are
            # done
            [queue.put(None) for _ in range(stop_signals)]
            break

def peek(records) :
    try :
        record = next(records)
    except TypeError as e:
        if "not an iterator" not in str(e) :
            raise
        try :
            records = iter(records)
            record = next(records)
        except StopIteration :
            return None, records
    except StopIteration :
        return None, records
    


    return record, itertools.chain([record], records)


def freezeData(data) : # pragma : no cover
    lfrozendict = frozendict
    return [(lfrozendict(record_1), 
             lfrozendict(record_2))
            for record_1, record_2 in data]

def isIndexed(data, offset) :
    hashable = collections.Hashable
    for i in range(offset, offset + len(data)) :
        if i not in data :
            return False
    else :
        return True

def index(data, offset=0) :
    if isIndexed(data, offset) :
        return data
    else :
        data = dict(zip(itertools.count(offset), 
                        viewvalues(data)))
        return data


class frozendict(collections.Mapping):
    """Don't forget the docstrings!!"""

    def __init__(self, arg): # pragma : no cover
        self._d = dict(arg)

    def __iter__(self):                  # pragma : no cover
        return iter(self._d)

    def __len__(self):                   # pragma : no cover
        return len(self._d)

    def __getitem__(self, key):          # pragma : no cover
        return self._d[key]

    def __repr__(self) :
        return u'<frozendict %s>' % repr(self._d)

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(frozenset(self._d.items()))
            return h


def cartesian(arrays, out=None): # pragma : no cover
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
    1-D arrays to form the cartesian product of.
    out : ndarray
    Array to place the cartesian product in.
    
    Returns
    -------
    out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.
    
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
    [1, 4, 7],
    [1, 5, 6],
    [1, 5, 7],
    [2, 4, 6],
    [2, 4, 7],
    [2, 5, 6],
    [2, 5, 7],
    [3, 4, 6],
    [3, 4, 7],
    [3, 5, 6],
    [3, 5, 7]])
    
    References
    ----------
    http://stackoverflow.com/q/1208118
    
    """
    arrays = [numpy.asarray(x).ravel() for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.empty([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out
