#!/usr/bin/python
# -*- coding: utf-8 -*-
# -*- coding: future_fstrings -*-

from collections import defaultdict
import time
import logging
from typing import Generator, Tuple, Iterable, Dict, List, Union
from dedupe._typing import Record, RecordID, Data
from dedupe.logger import logger
from timebudget import timebudget
import dedupe.predicates
logger.setLevel(logging.DEBUG)


Docs = Union[Iterable[str], Iterable[Iterable[str]]]


def index_list():
    return defaultdict(list)


class Fingerprinter(object):
    """Takes in a record and returns all blocks that record belongs to"""

    def __init__(self, predicates: Iterable[dedupe.predicates.Predicate]) -> None:
        """
        Args:
            predicates: (tuple)[dedupe.Predicate class] a tuple of predicates which
                are used to block the data. Can be a simple or compound predicate.
                For example, a tuple of compound predicates:
                ::
                    ((LevenshteinCanopyPredicate: (2, first_name),
                      SimplePredicate: (sortedAcronym, last_name)),
                     (SimplePredicate: (fingerprint, initials),
                      SimplePredicate: (nearIntegersPredicate, street_address)))

                where
                ::

                    (LevenshteinCanopyPredicate: (2, first_name),
                      SimplePredicate: (sortedAcronym, last_name))

                is a compound predicate (inherits from tuple)

                A tuple of simple predicates:
                ::
                    (SimplePredicate: (sortedAcronym, last_name),)

            index_fields: (dict) A dictionary of all the fingerprinter methods that use an
                index of data field values. The keys are the field names,
                which can be useful to know for indexing the data.

        """
        logger.debug("Initializing Fingerprinter class")
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

    # @timebudget
    def __call__(self,
                 records: Iterable[Record],
                 target: bool = False) -> Generator[Tuple[str, RecordID], None, None]:
        """Generate the predicates for records. Yields tuples of (predicate,
        record_id).

        Args:
            records: (dict_items) A sequence of tuples of (record_id,
                  record_dict). Can often be created by
                  `data_dict.items()`. List of input records.
                  key = (str) id of record
                  value = (dict) record
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

        Yields:
            A generator of tuples: ('predicate', 'record_id')

        Example:

            Say you only have 3 records:

        .. code:: python
            > data = [('0', {'first_name': 'John',
                             'last_name': 'Doe',
                             'city': 'Vancouver',
                             'gender': 'male',
                             'country': 'Canada'})
                      ('1', {'first_name': 'John',
                             'last_name': 'Donovan',
                             'city': 'Vancouver',
                             'gender': 'male',
                             'country': 'Canada'})
                      ('2', {'first_name': 'Alice',
                             'last_name': 'Walker',
                             'city': 'Victoria',
                             'gender': 'female',
                             'country': 'Canada'})]

        If your model has been trained already, you can see what predicates were chosen
        (these are just an example):
        .. code:: python
            > print(deduper.fingerprinter.predicates)
            ((SimplePredicate: (sortedAcronym, city),
              SimplePredicate: (sortedAcronym, gender)),
             (SimplePredicate: (wholeFieldPredicate, first_name),
             (SimplePredicate: (wholeFieldPredicate, last_name))))

        Your output would then be:
        .. code:: python
            > blocked_records = deduper.fingerprinter(data)
            > print(list(blocked_records))
            [('V:M:0', '0'),
             ('John:Doe:1', '0'),
             ('V:M:0', '1'),
             ('John:Donovan:1', '1'),
             ('V:F:0', '2'),
             ('Alice:Walker:1', '2')
            ]

        """
        # logger.debug("blocking.Fingerprinter.__call__")
        start_time = time.perf_counter()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]
        # logger.debug(f"Predicates: {predicates}")
        for i, record in enumerate(records):
            record_id, instance = record
            # logger.debug(f"Record: {record}")
            for pred_id, predicate in predicates:
                block_keys = predicate(instance, target=target)
                # logger.debug(f"Block keys: {block_keys}")
                for block_key in block_keys:
                    yield block_key + pred_id, record_id

            if i and i % 10000 == 0:
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                            {'iteration': i,
                             'elapsed': time.perf_counter() - start_time})

    def reset_indices(self) -> None:
        """
        Fingeprinter indicdes can take up a lot of memory. If you are
        done with blocking, the method will reset the indices to free up.
        If you need to block again, the data will need to be re-indexed.
        """
        for predicate in self.index_predicates:
            predicate.reset()

    def index(self, docs: Docs, field: str) -> None:
        """
        Add docs to the indices used by fingerprinters.
        Some fingerprinter methods depend upon having an index of
        values that a field may have in the data. This method adds
        those values to the index. If you don't have any fingerprinter
        methods that use an index, this method will do nothing.

        Args:
            docs: an iterator of values from your data to index. While
                  not required, it is recommended that docs be a unique
                  set of of those values. Indexing can be an expensive
                  operation.
            field: fieldname or key associated with the values you are
                   indexing
        """
        indices = extract_indices(self.index_fields[field])

        for doc in docs:
            if doc:
                for _, index, preprocess in indices:
                    index.index(preprocess(doc))

        for index_type, index, _ in indices:

            index.initSearch()
            for predicate in self.index_fields[field][index_type]:
                # logger.debug("Canopy: %s", str(predicate))
                predicate.reset()  # AH upgrade
                predicate.index = index

    def unindex(self, docs: Docs, field: str) -> None:
        """Remove docs from indices used by fingerprinters.

        Args:
            docs: an iterator of values from your data to remove. While
                  not required, it is recommended that docs be a unique
                  set of of those values. Indexing can be an expensive
                  operation.
            field: fieldname or key associated with the values you are
                   unindexing
        """

        indices = extract_indices(self.index_fields[field])

        for doc in docs:
            if doc:
                for _, index, preprocess in indices:
                    index.unindex(preprocess(doc))

        for index_type, index, _ in indices:

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                # logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def index_all(self, data: Data):
        for field in self.index_fields:
            unique_fields = {record[field]
                             for record
                             in data.values()
                             if record[field]}
            self.index(unique_fields, field)


def extract_indices(index_fields):
    indices = []
    for index_type, predicates in index_fields.items():
        predicate = predicates[0]
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None:
            index = predicate.initIndex()
        indices.append((index_type, index, preprocess))

    return indices
