#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import tempfile
import os
import collections
import warnings
import functools
import multiprocessing
import multiprocessing.dummy
import queue
from typing import (Iterator,
                    Tuple,
                    Mapping,
                    Sequence,
                    Union,
                    Generator,
                    Optional,
                    Any,
                    Type,
                    Iterable,
                    overload)

import numpy
from numpy.random import default_rng

from dedupe._typing import (RecordPairs,
                            RecordID,
                            Blocks,
                            Data,
                            Literal)
from dedupe.backport import RLock


rng = default_rng()

try:
    rng_integers = rng.integers  # type: ignore
except AttributeError:
    rng_integers = rng.randint  # type: ignore


class BlockingError(Exception):
    pass


_Queue = Union[multiprocessing.dummy.Queue, multiprocessing.Queue]
IndicesIterator = Iterator[Tuple[int, int]]


def randomPairs(n_records: int, sample_size: int) -> IndicesIterator:
    """
    Return random combinations of indices for a square matrix of size n
    records. For a discussion of how this works see
    http://stackoverflow.com/a/14839010/98080

    """
    n: int = n_records * (n_records - 1) // 2

    if sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        try:
            random_pairs = rng_integers(n, size=sample_size)
        except (OverflowError, ValueError):
            return randomPairsWithReplacement(n_records, sample_size)

    b: int = 1 - 2 * n_records

    i = (-b - 2 * numpy.sqrt(2 * (n - random_pairs) + 0.25)) // 2
    i = i.astype(int)

    j = random_pairs + i * (b + i + 2) // 2 + 1

    return zip(i, j)


def randomPairsMatch(n_records_A: int, n_records_B: int, sample_size: int) -> IndicesIterator:
    """
    Return random combinations of indices for record list A and B
    """
    n: int = n_records_A * n_records_B

    if sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        random_pairs = rng_integers(n, size=sample_size)

    i, j = numpy.unravel_index(random_pairs, (n_records_A, n_records_B))

    return zip(i, j)


def randomPairsWithReplacement(n_records: int, sample_size: int) -> IndicesIterator:
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("The same record pair may appear more than once in the sample")

    try:
        random_indices = rng_integers(n_records,
                                      size=sample_size * 2)
    except (OverflowError, ValueError):
        max_int: int = numpy.iinfo('int').max
        warnings.warn("Asked to sample pairs from %d records, will only sample pairs from first %d records" % (n_records, max_int))

        random_indices = rng_integers(max_int,
                                      size=sample_size * 2)

    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return ((p.item(), q.item()) for p, q in random_indices)


class ScoreDupes(object):
    def __init__(self,
                 data_model,
                 classifier,
                 records_queue: _Queue,
                 exception_queue: _Queue,
                 score_file_path: str,
                 dtype: numpy.dtype,
                 offset):
        self.data_model = data_model
        self.classifier = classifier
        self.records_queue = records_queue
        self.exception_queue = exception_queue
        self.score_file_path = score_file_path
        self.dtype = dtype
        self.offset = offset

    def __call__(self) -> None:

        while True:
            record_pairs: Optional[RecordPairs] = self.records_queue.get()
            if record_pairs is None:
                break

            try:
                self.fieldDistance(record_pairs)
            except Exception as e:
                self.exception_queue.put(e)
                raise

    def fieldDistance(self, record_pairs: RecordPairs) -> None:

        record_ids, records = zip(*(zip(*record_pair) for record_pair in record_pairs))

        if records:

            distances = self.data_model.distances(records)
            scores = self.classifier.predict_proba(distances)[:, -1]

            if scores.any():

                with self.offset.get_lock():

                    fp: numpy.memmap
                    fp = numpy.memmap(self.score_file_path,
                                      mode='r+',
                                      dtype=self.dtype,
                                      offset=self.offset.value,
                                      shape=(len(record_ids), ))
                    fp['pairs'] = record_ids
                    fp['score'] = scores

                    fp.flush()

                    self.offset.value += len(record_ids) * self.dtype.itemsize


def scoreDuplicates(record_pairs: RecordPairs,
                    data_model,
                    classifier,
                    num_cores: int = 1) -> Union[numpy.memmap, numpy.ndarray]:
    if num_cores < 2:
        from multiprocessing.dummy import Process, Queue
    else:
        from .backport import Process, Queue  # type: ignore

    first, record_pairs = peek(record_pairs)
    if first is None:
        raise BlockingError("No records have been blocked together. "
                            "Is the data you are trying to match like "
                            "the data you trained on? If so, try adding "
                            "more training data.")

    record_pairs_queue: _Queue = Queue(2)
    exception_queue: _Queue = Queue()
    scored_pairs_file, score_file_path = tempfile.mkstemp()
    os.close(scored_pairs_file)

    # explicitly defining the lock from the "spawn context" seems to
    # be necessary for python 3.7 on mac os.
    offset = multiprocessing.Value('Q', 0, lock=RLock())

    id_type = sniff_id_type(first)
    dtype = numpy.dtype([('pairs', id_type, 2),
                         ('score', 'f4')])

    n_map_processes = max(num_cores, 1)
    score_records = ScoreDupes(data_model,
                               classifier,
                               record_pairs_queue,
                               exception_queue,
                               score_file_path,
                               dtype,
                               offset)
    map_processes = [Process(target=score_records)
                     for _ in range(n_map_processes)]

    for process in map_processes:
        process.start()

    fillQueue(record_pairs_queue, record_pairs, n_map_processes)

    for process in map_processes:
        process.join()

    try:
        exc = exception_queue.get_nowait()
    except queue.Empty:
        pass
    else:
        raise ChildProcessError from exc

    scored_pairs: Union[numpy.memmap, numpy.ndarray]

    if offset.value:  # type: ignore
        scored_pairs = numpy.memmap(score_file_path,
                                    dtype=dtype)
    else:
        scored_pairs = numpy.array([], dtype=dtype)

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

        record_ids, records = zip(*(zip(*each) for each in block))

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
        return  # terminate iteration

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


class DiagonalEnumerator(object):
    def __init__(self, N: int):
        self.C = 2 * N - 3

    def __getitem__(self, pair: Tuple[int, int]) -> int:
        x, y = pair
        return x * (self.C - x) // 2 + y - 1


class FullEnumerator(object):
    def __init__(self, width: int):
        self.width = width

    def __getitem__(self, pair: Tuple[int, int]) -> int:
        x, y = pair
        return x * self.width + y


@overload
def sniff_id_type(ids: Sequence[Tuple[int, int]]) -> Type[int]:
    ...


@overload
def sniff_id_type(ids: Sequence[Tuple[str, str]]) -> Tuple[Type[str], int]:
    ...


def sniff_id_type(ids: Sequence[Tuple[RecordID, RecordID]]) -> Union[Type[int], Tuple[Type[str], int]]:
    example = ids[0][0]
    python_type = type(example)
    dtype: Union[Type[int], Tuple[Type[str], int]]
    if python_type is bytes or python_type is str:
        dtype = (str, 256)
    elif python_type is int:
        int(example)  # make sure we can cast to int
        dtype = int
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
