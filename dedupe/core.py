#!/usr/bin/python
# -*- coding: utf-8 -*-
# -*- coding: future_fstrings -*-

import itertools
import time
import tempfile
import os
import random
import collections
import warnings
import functools
import numpy
from dedupe.logger import logger
import multiprocessing
import multiprocessing.dummy
from typing import (Iterator,
                    Tuple,
                    Mapping,
                    Sequence,
                    Union,
                    Generator,
                    Optional,
                    Any,
                    Type,
                    Iterable)
from dedupe._typing import (RecordPairs, RecordID, Blocks, Data, Literal)


_Queue = Union[multiprocessing.dummy.Queue, multiprocessing.Queue]
_SimpleQueue = Union[multiprocessing.dummy.Queue, multiprocessing.SimpleQueue]
IndicesIterator = Iterator[Tuple[int, int]]


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

    def __init__(self,
                 distances,
                 classifier,
                 threshold,
                 records_queue: _Queue,
                 score_queue: _SimpleQueue):
        self.distances = distances
        self.classifier = classifier
        self.threshold = threshold
        self.records_queue = records_queue
        self.score_queue = score_queue

    def __call__(self) -> None:

        while True:
            record_pairs: Optional[RecordPairs] = self.records_queue.get()
            if record_pairs is None:
                break

            try:
                filtered_pairs: Optional[Tuple] = self.field_distance(record_pairs)
                if filtered_pairs is not None:
                    self.score_queue.put(filtered_pairs)
            except Exception as e:
                self.score_queue.put(e)
                raise

        self.score_queue.put(None)

    def field_distance(self, record_pairs: RecordPairs) -> Optional[Tuple]:
        """

        During the previous step, records were clustered (blocked) based on the
        predicates. For each proposed cluster, the records were combined
        pairwise, to create record_pairs.

        For example, suppose we have 5 records, and block them as follows:
            Block 1: record1, record2
            Block 2: record3, record4, record5

        The our record_pairs tuple would look like this:
            (
                (('id1', record1), ('id2', record2)),
                (('id3', record3), ('id4', record4)),
                (('id3', record3), ('id5', record5)),
                (('id4', record4), ('id5', record5))
            )

        Args:
            record_pairs: (tuple)[tuple]
                (
                    (('id1', record1, set()), ('id2', record2, set())),
                    (('id1', record1, set()), ('id3', record3, set()))

                )
        """
        logger.info("core.ScoreDupes.field_distance")
        record_ids, records = zip(*(zip(*record_pair) for record_pair in record_pairs))

        if records:
            distances = self.distances.compute_distance_matrix(records)
            scores = self.classifier.predict_proba(distances)[:, -1]
            mask = scores > self.threshold
            # logger.debug(distances)
            # logger.debug(scores)
            # logger.debug(f"Threshold = {self.threshold}")
            # logger.debug(f"Mask = {mask}")
            if mask.any():
                id_type = sniff_id_type(record_ids)
                ids = numpy.array(record_ids, dtype=id_type)

                dtype = numpy.dtype([('pairs', id_type, 2),
                                     ('score', 'f4')])

                temp_file, file_path = tempfile.mkstemp()
                os.close(temp_file)

                scored_pairs = numpy.memmap(file_path,
                                            shape=numpy.count_nonzero(mask),
                                            dtype=dtype)

                scored_pairs['pairs'] = ids[mask]
                scored_pairs['score'] = scores[mask]

                return file_path, dtype

        return None


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


def scoreDuplicates(records, distances, classifier, num_cores: int = 1, threshold=0):
    """
    Returns:
        scored_pairs: (generator) (np.array)[tuple(list[str], float)] A list of tuples,
            where each tuple contains an id pair and a probability that they are a match:
                id_pair_tuple: ([record_id_1, record_id_2], prob)
                dtype: np.dtype([('pairs', '<U256', 2),
                                 ('score', 'f4', 1)])

    Example:

        ..code:: python

            > scored_pairs = scoreDuplicates()
            > print(list(scored_pairs))
            > [(['19', '20'], 0.9990217 ),
                (['19', '21'], 0.9990217 ), (['19', '22'], 0.9990217 ),
                (['20', '21'], 0.9990217 ), (['20', '22'], 0.9990217 ),
                (['21', '22'], 0.9990217 ), (['22', '23'], 0.999513  )]
    """
    logger.info(f"Num cores: {num_cores}")
    num_cores = 1
    if num_cores < 2:
        from multiprocessing.dummy import Process, Queue
        SimpleQueue = Queue
    else:
        from .backport import Process, SimpleQueue, Queue

    first, records = peek(records)

    if first is None:
        return []

    record_pairs_queue = Queue(2)
    score_queue = SimpleQueue()
    result_queue = SimpleQueue()
    n_map_processes = max(num_cores, 1)
    score_records = ScoreDupes(distances,
                               classifier,
                               threshold,
                               record_pairs_queue,
                               score_queue)
    map_processes = [Process(target=score_records)
                     for _ in range(n_map_processes)]

    for process in map_processes:
        process.start()
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
    logger.debug(scored_pairs)
    return scored_pairs


# def fillQueue(queue, iterable, stop_signals):
#     iterable = iter(iterable)
#     chunk_size = 10000
#     upper_bound = 7000000  # this number worked, but is unprincipled
#     multiplier = 1.1
#
#     # initial values
#     i = 0
#     n_records = 0
#     t0 = time.perf_counter()
#     last_rate = 10000
#
#     while True:
#         chunk = tuple(itertools.islice(iterable, int(chunk_size)))
#         if chunk:
#             queue.put(chunk)
#             del chunk
#
#             n_records += chunk_size
#             i += 1
#
#             if i % 10:
#                 time_delta = max(time.perf_counter() - t0, 0.0001)
#
#                 current_rate = n_records / time_delta
#
#                 # chunk_size is always either growing or shrinking, if
#                 # the shrinking led to a faster rate, keep
#                 # shrinking. Same with growing. If the rate decreased,
#                 # reverse directions
#                 if current_rate < last_rate:
#                     multiplier = 1 / multiplier
#
#                 chunk_size = min(max(chunk_size * multiplier, 1), upper_bound)
#
#                 last_rate = current_rate
#                 n_records = 0
#                 t0 = time.perf_counter()
#
#         else:
#             # put poison pills in queue to tell scorers that they are
#             # done
#             [queue.put(None) for _ in range(stop_signals)]
#             break


def fillQueue(queue: _Queue,
              iterable: RecordPairs,
              stop_signals: int,
              chunk_size: int = 20000) -> None:
    iterable = iter(iterable)

    while True:
        chunk = tuple(itertools.islice(iterable, chunk_size))
        if chunk:
            queue.put(chunk)
            del chunk

        else:
            # put poison pills in queue to tell scorers that they are
            # done
            for _ in range(stop_signals):
                queue.put(None)
            break


class ScoreGazette(object):
    def __init__(self, distances, classifier, threshold):
        self.distances = distances
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

        distances = self.distances.compute_distance_matrix(records)
        scores = self.classifier.predict_proba(distances)[:, -1]

        mask = scores > self.threshold
        id_type = sniff_id_type(ids)
        ids = numpy.array(ids, dtype=id_type)

        dtype = numpy.dtype([('pairs', id_type, 2),
                             ('score', 'f4')])

        scored_pairs = numpy.empty(shape=numpy.count_nonzero(mask),
                                   dtype=dtype)

        scored_pairs['pairs'] = ids[mask]
        scored_pairs['score'] = scores[mask]

        return scored_pairs


def scoreGazette(records, distances, classifier, num_cores=1, threshold=0):

    first, records = peek(records)
    if first is None:
        raise ValueError("No records to match")

    imap, pool = appropriate_imap(num_cores)

    score_records = ScoreGazette(distances, classifier, threshold)

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
                        data.values()))
        return data


def Enumerator(start=0, initial=()):
    return collections.defaultdict(itertools.count(start).__next__, initial)


def sniff_id_type(ids):
    example = ids[0][0]
    python_type = type(example)
    if python_type is bytes or python_type is str:
        python_type = (str, 256)
    else:
        int(example)  # make sure we can cast to int
        python_type = int

    return python_type


def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned


def sqlite_id_type(data: Data) -> Literal['text', 'integer']:

    example = next(iter(data.keys()))
    python_type = type(example)

    if python_type is bytes or python_type is str:
        return 'text'
    elif python_type is int:
        return 'integer'
    else:
        raise ValueError('Invalid type for record id')
