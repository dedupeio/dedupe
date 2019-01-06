#!/usr/bin/python
# -*- coding: utf-8 -*-

from future.utils import viewvalues

from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


def index_list():
    return defaultdict(list)


class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''

    def __init__(self, predicates):

        self.predicates = predicates

        self.index_fields = defaultdict(index_list)
        self.index_predicates = []

        for full_predicate in predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index'):
                    self.index_fields[predicate.field][predicate.type].append(
                        predicate)
                    self.index_predicates.append(predicate)

    def __call__(self, records, target=False):

        start_time = time.clock()
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
                             'elapsed': time.clock() - start_time})

    def resetIndices(self):
        # clear canopies to reduce memory usage
        for predicate in self.index_predicates:
            predicate.reset()

    def index(self, data, field):
        '''Creates TF/IDF index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    index.index(preprocess(doc))

        for index_type, index, _ in indices:

            index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def unindex(self, data, field):
        '''Remove index of a given set of data'''
        indices = extractIndices(self.index_fields[field])

        for doc in data:
            if doc:
                for _, index, preprocess in indices:
                    index.unindex(preprocess(doc))

        for index_type, index, _ in indices:

            index._index.initSearch()

            for predicate in self.index_fields[field][index_type]:
                logger.debug("Canopy: %s", str(predicate))
                predicate.index = index

    def indexAll(self, data_d):
        for field in self.index_fields:
            unique_fields = {record[field]
                             for record
                             in viewvalues(data_d)
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
