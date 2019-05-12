#!/usr/bin/python
# -*- coding: utf-8 -*-
from builtins import range, next, zip, map
from future.utils import viewvalues

import sys
import itertools
import time
import tempfile
import os
import random
import collections
import warnings
import functools

import numpy

if sys.version < '3':
    text_type = unicode  # noqa: F821
    binary_type = str
    int_type = long  # noqa: F821
else:
    text_type = str
    binary_type = bytes
    unicode = str
    int_type = int


class ChildProcessError(Exception):
    pass


class BlockingError(Exception):
    pass


def randomPairs(n_records, sample_size):
    """
    Return random combinations of indices for a square matrix of size n
    records. For a discussion of how this works see
    http://stackoverflow.com/a/14839010/98080

    """
    n = int(n_records * (n_records - 1) / 2)

    if sample_size >= n:
        random_pairs = numpy.arange(n, dtype='uint')
    else:
        try:
            random_pairs = numpy.array(random.sample(range(n), sample_size),
                                       dtype='uint')
        except OverflowError:
            return randomPairsWithReplacement(n_records, sample_size)

    b = 1 - 2 * n_records

    root = (-b - 2 * numpy.sqrt(2 * (n - random_pairs) + 0.25)) / 2

    i = numpy.floor(root).astype('uint')
    j = numpy.rint(random_pairs + i * (b + i + 2) / 2 + 1).astype('uint')

    return zip(i, j)


def randomPairsMatch(n_records_A, n_records_B, sample_size):
    """
    Return random combinations of indices for record list A and B
    """
    n = int(n_records_A * n_records_B)

    if sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        random_pairs = numpy.array(random.sample(range(n), sample_size),
                                   dtype=int)

    i, j = numpy.unravel_index(random_pairs, (n_records_A, n_records_B))

    return zip(i, j)


def randomPairsWithReplacement(n_records, sample_size):
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("The same record pair may appear more than once in the sample")

    try:
        random_indices = numpy.random.randint(n_records,
                                              size=sample_size * 2)
    except (OverflowError, ValueError):
        max_int = numpy.iinfo('int').max
        warnings.warn("Asked to sample pairs from %d records, will only sample pairs from first %d records" % (n_records, max_int))

        random_indices = numpy.random.randint(max_int,
                                              size=sample_size * 2)

    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return [(p.item(), q.item()) for p, q in random_indices]


class ScoreDupes(object):
    def __init__(self, data_model, classifier, threshold):
        self.data_model = data_model
        self.classifier = classifier
        self.threshold = threshold
        self.score_queue = None

    def __call__(self, records_queue, score_queue):
        self.score_queue = score_queue
        while True:
            record_pairs = records_queue.get()
            if record_pairs is None:
                break

            try:
                filtered_pairs = self.fieldDistance(record_pairs)
                if filtered_pairs is not None:
                    score_queue.put(filtered_pairs)
            except Exception as e:
                score_queue.put(e)
                raise

        score_queue.put(None)

    def fieldDistance(self, record_pairs):
        ids = []
        records = []

        for record_pair in record_pairs:
            ((id_1, record_1, smaller_ids_1),
             (id_2, record_2, smaller_ids_2)) = record_pair

            if smaller_ids_1.isdisjoint(smaller_ids_2):

                ids.append((id_1, id_2))
                records.append((record_1, record_2))

        if records:

            distances = self.data_model.distances(records)
            scores = self.classifier.predict_proba(distances)[:, -1]

            mask = scores > self.threshold
            if mask.any():
                id_type = sniff_id_type(ids)
                ids = numpy.array(ids, dtype=id_type)

                dtype = numpy.dtype([('pairs', id_type, 2),
                                     ('score', 'f4', 1)])

                temp_file, file_path = tempfile.mkstemp()
                os.close(temp_file)

                scored_pairs = numpy.memmap(file_path,
                                            shape=numpy.count_nonzero(mask),
                                            dtype=dtype)

                scored_pairs['pairs'] = ids[mask]
                scored_pairs['score'] = scores[mask]

                return file_path, dtype


def mergeScores(score_queue, result_queue, stop_signals):
    scored_pairs_file, file_path = tempfile.mkstemp()
    os.close(scored_pairs_file)

    seen_signals = 0
    end = 0

    while seen_signals < stop_signals:

        score_chunk = score_queue.get()

        if isinstance(score_chunk, Exception):
            result_queue.put(score_chunk)
            raise
        elif score_chunk is None:
            seen_signals += 1
        else:
            score_file, dtype = score_chunk
            score_chunk = numpy.memmap(score_file, mode='r', dtype=dtype)

            chunk_size = len(score_chunk)

            fp = numpy.memmap(file_path, dtype=dtype,
                              offset=(end * dtype.itemsize),
                              shape=(chunk_size, ))

            fp[:chunk_size] = score_chunk

            end += chunk_size

            del score_chunk
            os.remove(score_file)

    if end:
        result_queue.put((file_path, dtype, end))
    else:
        result_queue.put(None)


def scoreDuplicates(records, data_model, classifier, num_cores=1, threshold=0):
    if num_cores < 2:
        from multiprocessing.dummy import Process, Queue
        SimpleQueue = Queue
    else:
        from .backport import Process, SimpleQueue, Queue

    first, records = peek(records)
    if first is None:
        raise BlockingError("No records have been blocked together. "
                            "Is the data you are trying to match like "
                            "the data you trained on?")

    record_pairs_queue = Queue(2)
    score_queue = SimpleQueue()
    result_queue = SimpleQueue()

    n_map_processes = max(num_cores, 1)
    score_records = ScoreDupes(data_model, classifier, threshold)
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
    if isinstance(result, Exception):
        raise ChildProcessError

    if result:
        scored_pairs_file, dtype, size = result
        scored_pairs = numpy.memmap(scored_pairs_file,
                                    dtype=dtype,
                                    shape=(size,))
    else:
        dtype = numpy.dtype([('pairs', object, 2),
                             ('score', 'f4', 1)])
        scored_pairs = numpy.array([], dtype=dtype)

    reduce_process.join()
    [process.join() for process in map_processes]

    return scored_pairs


def fillQueue(queue, iterable, stop_signals):
    iterable = iter(iterable)
    chunk_size = 10000
    upper_bound = 7000000  # this number worked, but is unprincipled
    multiplier = 1.1

    # initial values
    i = 0
    n_records = 0
    t0 = time.clock()
    last_rate = 10000

    while True:
        chunk = tuple(itertools.islice(iterable, int(chunk_size)))
        if chunk:
            queue.put(chunk)
            del chunk

            n_records += chunk_size
            i += 1

            if i % 10:
                time_delta = max(time.clock() - t0, 0.0001)

                current_rate = n_records / time_delta

                # chunk_size is always either growing or shrinking, if
                # the shrinking led to a faster rate, keep
                # shrinking. Same with growing. If the rate decreased,
                # reverse directions
                if current_rate < last_rate:
                    multiplier = 1 / multiplier

                chunk_size = min(max(chunk_size * multiplier, 1), upper_bound)

                last_rate = current_rate
                n_records = 0
                t0 = time.clock()

        else:
            # put poison pills in queue to tell scorers that they are
            # done
            [queue.put(None) for _ in range(stop_signals)]
            break


class ScoreGazette(object):
    def __init__(self, data_model, classifier, threshold):
        self.data_model = data_model
        self.classifier = classifier
        self.threshold = threshold

    def __call__(self, block):
        ids = []
        records = []

        for record_pair in block:
            ((id_1, record_1, _),
             (id_2, record_2, _)) = record_pair

            ids.append((id_1, id_2))
            records.append((record_1, record_2))

        distances = self.data_model.distances(records)
        scores = self.classifier.predict_proba(distances)[:, -1]

        mask = scores > self.threshold
        id_type = sniff_id_type(ids)
        ids = numpy.array(ids, dtype=id_type)

        dtype = numpy.dtype([('pairs', id_type, 2),
                             ('score', 'f4', 1)])

        scored_pairs = numpy.empty(shape=numpy.count_nonzero(mask),
                                   dtype=dtype)

        scored_pairs['pairs'] = ids[mask]
        scored_pairs['score'] = scores[mask]

        return scored_pairs


def scoreGazette(records, data_model, classifier, num_cores=1, threshold=0):

    first, records = peek(records)
    if first is None:
        raise ValueError("No records to match")

    if sys.version < '3':
        records = (list(y) for y in records)

    imap, pool = appropriate_imap(num_cores)

    score_records = ScoreGazette(data_model, classifier, threshold)

    for scored_pairs in imap(score_records, records):
        yield scored_pairs

    # The underlying processes in the pool should terminate when the
    # pool is garbage collected, but sometimes it takes a while
    # before GC, so do it explicitly here
    pool.close()
    pool.join()


def appropriate_imap(num_cores):
    if num_cores < 2:
        imap = map

        # in order to make it simpler to cleanup a pool of processes
        # always return something that we can close and join
        class MockPool(object):
            def close(self):
                pass

            def join(self):
                pass
        pool = MockPool()
    else:
        from .backport import Pool
        pool = Pool(processes=num_cores)
        imap = functools.partial(pool.imap_unordered, chunksize=1)

    return imap, pool


def peek(records):
    try:
        record = next(records)
    except TypeError as e:
        if "not an iterator" not in str(e):
            raise
        try:
            records = iter(records)
            record = next(records)
        except StopIteration:
            return None, records
    except StopIteration:
        return None, records

    return record, itertools.chain([record], records)


def isIndexed(data, offset):
    return all(i in data for i in range(offset, offset + len(data)))


def index(data, offset=0):
    if isIndexed(data, offset):
        return data
    else:
        data = dict(zip(itertools.count(offset),
                        viewvalues(data)))
        return data


def Enumerator(start=0, initial=()):
    try:  # py 2
        return collections.defaultdict(itertools.count(start).next, initial)
    except AttributeError:  # py 3
        return collections.defaultdict(itertools.count(start).__next__, initial)


def sniff_id_type(ids):
    example = ids[0][0]
    python_type = type(example)
    if python_type is binary_type or python_type is text_type:
        python_type = (unicode, 256)
    else:
        int_type(example)  # make sure we can cast to int
        python_type = int_type

    return python_type
