#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import tempfile
import os
import random
import collections
import warnings
import functools

from typing import (Iterator,
                    Tuple,
                    Mapping,
                    Sequence,
                    Union,
                    Generator,
                    Optional,
                    Any,
                    Type,
                    Iterable, cast)
from dedupe._typing import (RecordPairs,
                            RecordID,
                            RecordDict,
                            Blocks,
                            Data,
                            Literal)

import numpy
import multiprocessing
import multiprocessing.dummy


class ChildProcessError(Exception):
    pass


class BlockingError(Exception):
    pass


_Queue = Union[multiprocessing.dummy.Queue, multiprocessing.Queue]
_SimpleQueue = Union[multiprocessing.dummy.Queue, multiprocessing.SimpleQueue]
IndicesIterator = Iterator[Tuple[int, int]]


def randomPairs(n_records: int, sample_size: int) -> IndicesIterator:
    """
    Return random combinations of indices for a square matrix of size n
    records. For a discussion of how this works see
    http://stackoverflow.com/a/14839010/98080

    """
    n: int = int(n_records * (n_records - 1) / 2)

    if sample_size >= n:
        random_pairs = numpy.arange(n, dtype='uint')
    else:
        try:
            random_pairs = numpy.array(random.sample(range(n), sample_size),
                                       dtype='uint')
        except OverflowError:
            return randomPairsWithReplacement(n_records, sample_size)

    b: int = 1 - 2 * n_records

    root = (-b - 2 * numpy.sqrt(2 * (n - random_pairs) + 0.25)) / 2

    i = numpy.floor(root).astype('uint')
    j = numpy.rint(random_pairs + i * (b + i + 2) / 2 + 1).astype('uint')

    return zip(i, j)


def randomPairsMatch(n_records_A: int, n_records_B: int, sample_size: int) -> IndicesIterator:
    """
    Return random combinations of indices for record list A and B
    """
    n: int = int(n_records_A * n_records_B)

    if sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        random_pairs = numpy.array(random.sample(range(n), sample_size),
                                   dtype=int)

    i, j = numpy.unravel_index(random_pairs, (n_records_A, n_records_B))

    return zip(i, j)


def randomPairsWithReplacement(n_records: int, sample_size: int) -> IndicesIterator:
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("The same record pair may appear more than once in the sample")

    try:
        random_indices = numpy.random.randint(n_records,
                                              size=sample_size * 2)
    except (OverflowError, ValueError):
        max_int: int = numpy.iinfo('int').max
        warnings.warn("Asked to sample pairs from %d records, will only sample pairs from first %d records" % (n_records, max_int))

        random_indices = numpy.random.randint(max_int,
                                              size=sample_size * 2)

    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return ((p.item(), q.item()) for p, q in random_indices)


class ScoreDupes(object):
    def __init__(self,
                 data_model,
                 classifier,
                 records_queue: _Queue,
                 score_queue: _SimpleQueue):
        self.data_model = data_model
        self.classifier = classifier
        self.records_queue = records_queue
        self.score_queue = score_queue

    def __call__(self) -> None:

        while True:
            record_pairs: Optional[RecordPairs] = self.records_queue.get()
            if record_pairs is None:
                break

            try:
                filtered_pairs: Optional[Tuple] = self.fieldDistance(record_pairs)
                if filtered_pairs is not None:
                    self.score_queue.put(filtered_pairs)
            except Exception as e:
                self.score_queue.put(e)
                raise

        self.score_queue.put(None)

    def fieldDistance(self, record_pairs: RecordPairs) -> Optional[Tuple]:

        record_ids, records = zip(*(zip(*record_pair) for record_pair in record_pairs))  # type: ignore
        record_ids = cast(Tuple[Tuple[RecordID, RecordID], ...], record_ids)
        records = cast(Tuple[Tuple[RecordDict, RecordDict], ...], records)

        if records:

            distances = self.data_model.distances(records)
            scores = self.classifier.predict_proba(distances)[:, -1]

            if scores.any():
                id_type = sniff_id_type(record_ids)
                ids = numpy.array(record_ids, dtype=id_type)

                dtype = numpy.dtype([('pairs', id_type, 2),
                                     ('score', 'f4')])

                temp_file, file_path = tempfile.mkstemp()
                os.close(temp_file)

                scored_pairs = numpy.memmap(file_path,
                                            shape=len(scores),
                                            dtype=dtype)

                scored_pairs['pairs'] = ids
                scored_pairs['score'] = scores

                return file_path, dtype

        return None


def mergeScores(score_queue: _SimpleQueue,
                result_queue: _SimpleQueue,
                stop_signals: int):
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


def scoreDuplicates(record_pairs: RecordPairs,
                    data_model,
                    classifier,
                    num_cores: int = 1):
    if num_cores < 2:
        from multiprocessing.dummy import Process, Queue
        SimpleQueue = Queue
    else:
        from .backport import Process, SimpleQueue, Queue  # type: ignore

    first, record_pairs = peek(record_pairs)
    if first is None:
        raise BlockingError("No records have been blocked together. "
                            "Is the data you are trying to match like "
                            "the data you trained on?")

    record_pairs_queue: _Queue = Queue(2)
    score_queue: _SimpleQueue = SimpleQueue()
    result_queue: _SimpleQueue = SimpleQueue()

    n_map_processes = max(num_cores, 1)
    score_records = ScoreDupes(data_model,
                               classifier,
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

    fillQueue(record_pairs_queue, record_pairs, n_map_processes)

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

    for process in map_processes:
        process.join()

    return scored_pairs


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
    def __init__(self, data_model, classifier):
        self.data_model = data_model
        self.classifier = classifier

    def __call__(self, block: RecordPairs) -> numpy.ndarray:

        record_ids, records = zip(*(zip(*each) for each in block))  # type: ignore
        record_ids = cast(Tuple[Tuple[RecordID, RecordID], ...], record_ids)
        records = cast(Tuple[Tuple[RecordDict, RecordDict], ...], records)

        distances = self.data_model.distances(records)
        scores = self.classifier.predict_proba(distances)[:, -1]

        id_type = sniff_id_type(record_ids)
        ids = numpy.array(record_ids, dtype=id_type)

        dtype = numpy.dtype([('pairs', id_type, 2),
                             ('score', 'f4')])

        scored_pairs = numpy.empty(shape=len(scores),
                                   dtype=dtype)

        scored_pairs['pairs'] = ids
        scored_pairs['score'] = scores

        return scored_pairs


def scoreGazette(record_pairs: Blocks,
                 data_model,
                 classifier,
                 num_cores: int = 1) -> Generator[numpy.ndarray, None, None]:

    first, record_pairs = peek(record_pairs)
    if first is None:
        raise ValueError("No records to match")

    imap, pool = appropriate_imap(num_cores)

    score_records = ScoreGazette(data_model, classifier)

    for scored_pairs in imap(score_records, record_pairs):
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
        imap = functools.partial(pool.imap_unordered, chunksize=20000)

    return imap, pool


def peek(seq: Iterator) -> Tuple[Optional[Any], Iterator]:
    try:
        first = next(seq)
    except TypeError as e:
        if "not an iterator" not in str(e):
            raise
        try:
            seq = iter(seq)
            first = next(seq)
        except StopIteration:
            return None, iter(seq)
    except StopIteration:
        return None, iter(seq)

    return first, itertools.chain([first], seq)


def isIndexed(data: Mapping, offset: int) -> bool:
    return all(i in data for i in range(offset, offset + len(data)))


def index(data: Mapping[Any, Any], offset: int = 0) -> Mapping[int, Any]:
    if isIndexed(data, offset):
        return data
    else:
        data = dict(zip(itertools.count(offset),
                        data.values()))
        return data


def Enumerator(start: int = 0, initial: tuple = ()) -> collections.defaultdict:
    return collections.defaultdict(itertools.count(start).__next__, initial)


def sniff_id_type(ids: Sequence[Tuple[RecordID, RecordID]]) -> Union[Type[int], Tuple[Type[str], int]]:
    example = ids[0][0]
    python_type = type(example)
    if python_type is bytes or python_type is str:
        dtype: Union[Type[int], Tuple[Type[str], int]] = (str, 256)
    elif python_type is int:
        int(example)  # make sure we can cast to int
        dtype: Union[Type[int], Tuple[Type[str], int]] = int  # type: ignore
    else:
        raise ValueError('Invalid type for record id')

    return dtype


def sqlite_id_type(data: Data) -> Literal['text', 'integer']:

    example = next(iter(data.keys()))
    python_type = type(example)

    if python_type is bytes or python_type is str:
        return 'text'
    elif python_type is int:
        return 'integer'
    else:
        raise ValueError('Invalid type for record id')


def unique(seq: Iterable) -> list:
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned: list = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned
