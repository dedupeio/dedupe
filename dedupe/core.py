#!/usr/bin/python
# -*- coding: utf-8 -*-
from builtins import range, next, zip, map
from future.utils import viewvalues, viewitems
import sys
if sys.version < '3':
    text_type = unicode
    binary_type = str
else:
    text_type = str
    binary_type = bytes
    unicode = str

import itertools
import time
import tempfile
import os
import operator
import random

import numpy

import dedupe.backport as backport

class ChildProcessError(Exception) :
    pass

def randomPairs(n_records, sample_size):
    """
    Return random combinations of indices for a square matrix of size
    n records
    """
    n = int(n_records * (n_records - 1) / 2)

    if sample_size >= n :
        random_pairs = numpy.arange(n, dtype='f')
    else:
        random_pairs = numpy.array(random.sample(range(n), sample_size), dtype='f')
    
    b = 1 - 2 * n_records

    i = numpy.floor((-b - numpy.sqrt(b ** 2 - 8 * random_pairs)) / 2).astype('uint')
    j = numpy.rint(random_pairs + i * (b + i + 2) / 2 + 1).astype('uint')

    return zip(i, j)

def randomPairsMatch(n_records_A, n_records_B, sample_size):
    """
    Return random combinations of indices for record list A and B
    """
    n = int(n_records_A * n_records_B)

    if sample_size >= n:
        random_pairs = numpy.arange(n, dtype='f')
    else:
        random_pairs = numpy.array(random.sample(range(n), sample_size), dtype='f')

    i = numpy.floor(random_pairs/n_records_B).astype('uint')
    j = (random_pairs - n_records_B * i).astype('uint')

    return zip(i, j)

class ScoreRecords(object) :
    def __init__(self, data_model, classifier, threshold) :
        self.data_model = data_model
        self.classifier = classifier
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

            if smaller_ids_1.isdisjoint(smaller_ids_2) :
                
                ids.append((id_1, id_2))
                records.append((record_1, record_2))

        if records :
            ids = numpy.array(ids)
            
            distances = self.data_model.distances(records)
            scores = self.classifier.predict_proba(distances)[:,-1]

            mask = scores > self.threshold

            scored_pairs = numpy.empty(numpy.count_nonzero(mask),
                                       dtype=[('pairs', 'object', 2), 
                                              ('score', 'f4', 1)])
            scored_pairs['pairs'] = ids[mask]
            scored_pairs['score'] = scores[mask]

            return scored_pairs

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

def scoreDuplicates(records, data_model, classifier, num_cores=1, threshold=0) :
    if num_cores < 2 :
        from multiprocessing.dummy import Process, Pool, Queue
        SimpleQueue = Queue
    else :
        from .backport import Process, Pool, SimpleQueue

    record_pairs_queue = SimpleQueue()
    score_queue =  SimpleQueue()
    result_queue = SimpleQueue()

    n_map_processes = max(num_cores-1, 1)
    score_records = ScoreRecords(data_model, classifier, threshold) 
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

    reduce_process.join()
    [process.join() for process in map_processes]

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


def isIndexed(data, offset) :
    return all(i in data for i in range(offset, offset + len(data)))

def index(data, offset=0) :
    if isIndexed(data, offset):
        return data
    else :
        data = dict(zip(itertools.count(offset), 
                        viewvalues(data)))
        return data

def iunzip(iterable, internal_length): # pragma: no cover
    """Iunzip is the same as zip(*iter) but returns iterators, instead of 
    expand the iterator. Mostly used for large sequence"""

    _tmp, iterable = itertools.tee(iterable, 2)
    iters = itertools.tee(iterable, internal_length)
    return (map(operator.itemgetter(i), it) for i, it in enumerate(iters))

