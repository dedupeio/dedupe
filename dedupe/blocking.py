#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import logging
import time

from typing import Generator, Tuple, Iterable, Dict, List, Union
from dedupe._typing import Record, RecordID, Data

import dedupe.predicates

logger = logging.getLogger(__name__)


def index_list():
    return defaultdict(list)


class Fingerprinter(object):
    '''Takes in a record and returns all blocks that record belongs to'''

    def __init__(self, predicates: Iterable[dedupe.predicates.Predicate]) -> None:

        self.predicates = predicates

        self.index_fields: Dict[str,
                                Dict[str,
                                     List[dedupe.predicates.IndexPredicate]]]
        self.index_fields = defaultdict(index_list)

        self.index_predicates = []

        for full_predicate in predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index'):
                    self.index_fields[predicate.field][predicate.type].append(
                        predicate)
                    self.index_predicates.append(predicate)

    def __call__(self,
                 records: Iterable[Record],
                 target: bool = False) -> Generator[Tuple[str, RecordID], None, None]:
        '''
        Generate the predicates for records. Yields tuples of (predicate,
        record_id).

        Args:
            records: A sequence of tuples of (record_id,
                  record_dict). Can often be created by
                  `data_dict.items()`.
            target: Indicates whether the data should be treated as
                    the target data. This effects the behavior of
                    search predicates. If `target` is set to
                    `True`, an search predicate will return the
                    value itself. If `target` is set to `False` the
                    search predicate will return all possible
                    values within the specified search distance.

                    Let's say we have a
                    `LevenshteinSearchPredicate` with an associated
                    distance of `1` on a `"name"` field; and we
                    have a record like `{"name": "thomas"}`. If the
                    `target` is set to `True` then the predicate
                    will return `"thomas"`.  If `target` is set to
                    `False`, then the blocker could return
                    `"thomas"`, `"tomas"`, and `"thoms"`. By using
                    the `target` argument on one of your datasets,
                    you will dramatically reduce the total number
                    of comparisons without a loss of accuracy.
        '''

        start_time = time.perf_counter()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        for i, record in enumerate(records):
            record_id, instance = record

            for pred_id, predicate in predicates:
                block_keys = predicate(instance, target=target)
                for block_key in block_keys:
                    yield block_key + pred_id, record_id

            if i and i % 10000 == 0:
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                            {'iteration': i,
                             'elapsed': time.perf_counter() - start_time})

    def reset_indices(self):
        # clear canopies to reduce memory usage
        for predicate in self.index_predicates:
            predicate.reset()

    def index(self,
              data: Union[Iterable[str], Iterable[Iterable[str]]],
              field: str):
        '''Creates index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    index.index(preprocess(doc))

        for index_type, index, _ in indices:

            index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.reset()
                predicate.index = index

    def unindex(self, data: Iterable, field: str):
        '''Remove index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    try:
                        index.unindex(preprocess(doc))
                    except KeyError:
                        pass

        for index_type, index, _ in indices:

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def index_all(self, data_d: Data):
        for field in self.index_fields:
            unique_fields = {record[field]
                             for record
                             in data_d.values()
                             if record[field]}
            self.index(unique_fields, field)


def extractIndices(index_fields):

    indices = []
    for index_type, predicates in index_fields.items():
        predicate = predicates[0]
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None:
            index = predicate.initIndex()
        indices.append((index_type, index, preprocess))

    return indices
