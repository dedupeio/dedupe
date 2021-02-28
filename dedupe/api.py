#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""

import itertools
import logging
import pickle
import multiprocessing
import warnings
import os
import sqlite3
import tempfile

import numpy
import json
import rlr

import dedupe.core as core
import dedupe.serializer as serializer
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.datamodel as datamodel
import dedupe.labeler as labeler
import dedupe.predicates

from typing import (Mapping,
                    Optional,
                    List,
                    Tuple,
                    Set,
                    Dict,
                    Union,
                    Generator,
                    Iterable,
                    Sequence,
                    BinaryIO,
                    cast,
                    TextIO)
from typing_extensions import Literal
from dedupe._typing import (Data,
                            Clusters,
                            RecordPairs,
                            RecordID,
                            RecordDict,
                            Blocks,
                            TrainingExample,
                            LookupResults,
                            Links,
                            TrainingData,
                            Classifier,
                            JoinConstraint)

logger = logging.getLogger(__name__)


class Matching(object):
    """
    Base Class for Record Matching Classes
    """

    def __init__(self, num_cores: Optional[int], **kwargs) -> None:

        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self._fingerprinter: Optional[blocking.Fingerprinter] = None
        self.data_model: datamodel.DataModel
        self.classifier: Classifier
        self.predicates: List[dedupe.predicates.Predicate]

    @property
    def fingerprinter(self) -> blocking.Fingerprinter:

        if self._fingerprinter is None:
            raise ValueError('the record fingerprinter is not intialized, '
                             'please run the train method')

        return self._fingerprinter


class IntegralMatching(Matching):
    """
    This class is for linking class where we need to score all possible
    pairs before deciding on any matches
    """

    def score(self,
              pairs: RecordPairs) -> numpy.memmap:
        """
        Scores pairs of records. Returns pairs of tuples of records id and
        associated probabilites that the pair of records are match

        Args:
            pairs: Iterator of pairs of records

        """
        try:
            matches = core.scoreDuplicates(pairs,
                                           self.data_model,
                                           self.classifier,
                                           self.num_cores)
        except RuntimeError:
            raise RuntimeError('''
                You need to either turn off multiprocessing or protect
                the calls to the Dedupe methods with a
                `if __name__ == '__main__'` in your main module, see
                https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods''')

        return matches


class DedupeMatching(IntegralMatching):
    """
    Class for Deduplication, extends Matching.

    Use DedupeMatching when you have a dataset that can contain
    multiple references to the same entity.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def partition(self,
                  data: Data,
                  threshold: float = 0.5) -> Clusters:  # pragma: no cover
        """
        Identifies records that all refer to the same entity, returns
        tuples containing a sequence of record ids and corresponding
        sequence of confidence score as a float between 0 and 1. The
        record_ids within each set should refer to the same entity and the
        confidence score is a measure of our confidence a particular entity
        belongs in the cluster.

        For details on the confidence score, see :func:`dedupe.Dedupe.cluster`.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to :func:`~score`.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

            threshold: Number between 0 and 1 (Default is 0.5).  We
                       will only consider put together records into
                       clusters if the `cophenetic similarity
                       <https://en.wikipedia.org/wiki/Cophenetic>`_ of
                       the cluster is greater than the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

        .. code:: python

           > clusters = matcher.partition(data, threshold=0.5)
           > print(duplicates)
           [((1, 2, 3), (0.790, 0.860, 0.790)),
            ((4, 5), (0.720, 0.720)),
            ((10, 11), (0.899, 0.899))]

        """
        pairs = self.pairs(data)
        pair_scores = self.score(pairs)
        clusters = self.cluster(pair_scores, threshold)

        clusters = self._add_singletons(data, clusters)

        clusters = list(clusters)

        try:
            mmap_file = pair_scores.filename
            del pair_scores
            os.remove(mmap_file)
        except AttributeError:
            pass

        return clusters

    def _add_singletons(self, data, clusters):

        singletons = set(data.keys())

        for record_ids, score in clusters:
            singletons.difference_update(record_ids)
            yield (record_ids, score)

        for singleton in singletons:
            yield (singleton, ), (1.0, )

    def pairs(self, data):
        '''
        Yield pairs of records that share common fingerprints.

        Each pair will occur at most once. If you override this
        method, you need to take care to ensure that this remains
        true, as downstream methods, particularly :func:`cluster`, assumes
        that every pair of records is compared no more than once.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

        .. code:: python

            > pairs = matcher.pairs(data)
            > print(list(pairs))
            [((1, {'name' : 'Pat', 'address' : '123 Main'}),
              (2, {'name' : 'Pat', 'address' : '123 Main'})),
             ((1, {'name' : 'Pat', 'address' : '123 Main'}),
              (3, {'name' : 'Sam', 'address' : '123 Main'}))
             ]

        '''

        self.fingerprinter.index_all(data)

        id_type = core.sqlite_id_type(data)

        # Blocking and pair generation are typically the first memory
        # bottlenecks, so we'll use sqlite3 to avoid doing them in memory
        with tempfile.TemporaryDirectory() as temp_dir:
            con = sqlite3.connect(temp_dir + '/blocks.db')

            # Set journal mode to WAL.
            con.execute('pragma journal_mode=wal')

            con.execute('''CREATE TABLE blocking_map
                           (block_key text, record_id {id_type})
                        '''.format(id_type=id_type))

            con.executemany("INSERT INTO blocking_map values (?, ?)",
                            self.fingerprinter(data.items()))

            self.fingerprinter.reset_indices()

            con.execute('''CREATE INDEX block_key_idx
                           ON blocking_map (block_key)''')
            pairs = con.execute('''SELECT DISTINCT a.record_id, b.record_id
                                   FROM blocking_map a
                                   INNER JOIN blocking_map b
                                   USING (block_key)
                                   WHERE a.record_id < b.record_id''')

            for a_record_id, b_record_id in pairs:
                yield ((a_record_id, data[a_record_id]),
                       (b_record_id, data[b_record_id]))

            pairs.close()
            con.close()

    def cluster(self,
                scores: numpy.ndarray,
                threshold: float = 0.5) -> Clusters:
        r"""From the similarity scores of pairs of records, decide which groups
        of records are all referring to the same entity.

        Yields tuples containing a sequence of record ids and corresponding
        sequence of confidence score as a float between 0 and 1. The
        record_ids within each set should refer to the same entity and the
        confidence score is a measure of our confidence a particular entity
        belongs in the cluster.

        Each confidence scores is a measure of how similar the record is
        to the other records in the cluster. Let :math:`\phi(i,j)` be the pair-wise
        similarity between records :math:`i` and :math:`j`. Let :math:`N` be the number of records in the cluster.

        .. math::

           \text{confidence score}_i = 1 - \sqrt {\frac{\sum_{j}^N (1 - \phi(i,j))^2}{N -1}}

        This measure is similar to the average squared distance
        between the focal record and the other records in the
        cluster. These scores can be `combined to give a total score
        for the cluster
        <https://en.wikipedia.org/wiki/Variance#Discrete_random_variable>`_.

        .. math::

           \text{cluster score} = 1 - \sqrt { \frac{\sum_i^N(1 - \mathrm{score}_i)^2 \cdot (N - 1) } { 2 N^2}}

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

                    For each pair, the smaller id should be first.

            threshold: Number between 0 and 1. We will only consider
                       put together records into clusters if the
                       `cophenetic similarity
                       <https://en.wikipedia.org/wiki/Cophenetic>`_ of
                       the cluster is greater than the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

                       Defaults to 0.5.

        .. code:: python

           > pairs = matcher.pairs(data)
           > scores = matcher.scores(pairs)
           > clusters = matcher.cluster(scores)
           > print(list(clusters))
           [((1, 2, 3), (0.790, 0.860, 0.790)),
            ((4, 5), (0.720, 0.720)),
            ((10, 11), (0.899, 0.899))]

        """

        logger.debug("matching done, begin clustering")

        yield from clustering.cluster(scores, threshold)


class RecordLinkMatching(IntegralMatching):
    """
    Class for Record Linkage, extends Matching.

    Use RecordLinkMatching when you have two datasets that you want to merge
    """

    def pairs(self, data_1: Data, data_2: Data) -> RecordPairs:
        """
        Yield pairs of records that share common fingerprints.

        Each pair will occur at most once. If you override this
        method, you need to take care to ensure that this remains
        true, as downstream methods, particularly :func:`one_to_one`,
        and :func:`many_to_one` assumes that every pair of records is
        compared no more than once.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names
            data_2: Dictionary of records from second dataset, same
                    form as data_1

        .. code:: python

           > pairs = matcher.pairs(data_1, data_2)
           > print(list(pairs))
           [((1, {'name' : 'Pat', 'address' : '123 Main'}),
             (2, {'name' : 'Pat', 'address' : '123 Main'})),
            ((1, {'name' : 'Pat', 'address' : '123 Main'}),
             (3, {'name' : 'Sam', 'address' : '123 Main'}))
            ]
        """

        self.fingerprinter.index_all(data_2)

        id_type_a = core.sqlite_id_type(data_1)
        id_type_b = core.sqlite_id_type(data_2)

        # Blocking and pair generation are typically the first memory
        # bottlenecks, so we'll use sqlite3 to avoid doing them in memory
        with tempfile.TemporaryDirectory() as temp_dir:
            con = sqlite3.connect(temp_dir + '/blocks.db')

            # Set journal mode to WAL.
            con.execute('pragma journal_mode=wal')

            con.executescript('''CREATE TABLE blocking_map_a
                                 (block_key text, record_id {id_type_a});

                                 CREATE TABLE blocking_map_b
                                 (block_key text, record_id {id_type_b});
                              '''.format(id_type_a=id_type_a,
                                         id_type_b=id_type_b))

            con.executemany("INSERT INTO blocking_map_a values (?, ?)",
                            self.fingerprinter(data_1.items()))

            con.executemany("INSERT INTO blocking_map_b values (?, ?)",
                            self.fingerprinter(data_2.items(), target=True))

            self.fingerprinter.reset_indices()

            con.executescript('''CREATE INDEX block_key_a_idx
                                 ON blocking_map_a (block_key);

                                 CREATE INDEX block_key_b_idx
                                 ON blocking_map_b (block_key);''')

            pairs = con.execute('''SELECT DISTINCT a.record_id, b.record_id
                                   FROM blocking_map_a a
                                   INNER JOIN blocking_map_b b
                                   USING (block_key)''')

            for a_record_id, b_record_id in pairs:
                yield ((a_record_id, data_1[a_record_id]),
                       (b_record_id, data_2[b_record_id]))

            pairs.close()
            con.close()

    def join(self,
             data_1: Data,
             data_2: Data,
             threshold: float = 0.5,
             constraint: JoinConstraint = "one-to-one") -> Links:
        """
        Identifies pairs of records that refer to the same entity.

        Returns pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to the :func:`~score`.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names

            data_2: Dictionary of records from second dataset, same form
                    as data_1

            threshold: Number between 0 and 1 (default is .5). We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

            constraint: What type of constraint to put on a join.

                        'one-to-one'
                              Every record in data_1 can match at most
                              one record from data_2 and every record
                              from data_2 can match at most one record
                              from data_1. This is good for when both
                              data_1 and data_2 are from different
                              sources and you are interested in
                              matching across the sources. If,
                              individually, data_1 or data_2 have many
                              duplicates you will not get good
                              matches.
                        'many-to-one'
                              Every record in data_1 can match at most
                              one record from data_2, but more than
                              one record from data_1 can match to the
                              same record in data_2. This is good for
                              when data_2 is a lookup table and data_1
                              is messy, such as geocoding or matching
                              against golden records.
                        'many-to-many'
                              Every record in data_1 can match
                              multiple records in data_2 and vice
                              versa. This is like a SQL inner join.

        .. code:: python

           > links = matcher.join(data_1, data_2, threshold=0.5)
           > print(list(links))
           [((1, 2), 0.790),
            ((4, 5), 0.720),
            ((10, 11), 0.899)]



        """

        assert constraint in {'one-to-one', 'many-to-one', 'many-to-many'}, (
            '%s is an invalid constraint option. Valid options include '
            'one-to-one, many-to-one, or many-to-many' % constraint)

        pairs = self.pairs(data_1, data_2)
        pair_scores = self.score(pairs)

        if constraint == 'one-to-one':
            links = self.one_to_one(pair_scores, threshold)
        elif constraint == 'many-to-one':
            links = self.many_to_one(pair_scores, threshold)
        else:
            links = pair_scores[pair_scores['score'] > threshold]

        links = list(links)

        try:
            mmap_file = pair_scores.filename
            del pair_scores
            os.remove(mmap_file)
        except AttributeError:
            pass

        return links

    def one_to_one(self,
                   scores: numpy.ndarray,
                   threshold: float = 0.0) -> Links:
        """From the similarity scores of pairs of records, decide which
        pairs refer to the same entity.

        Every record in data_1 can match at most one record from
        data_2 and every record from data_2 can match at most one
        record from data_1. See
        https://en.wikipedia.org/wiki/Injective_function.

        This method is good for when both data_1 and data_2 are from
        different sources and you are interested in matching across
        the sources. If, individually, data_1 or data_2 have many duplicates
        you will not get good matches.

        Yields pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

            threshold: Number between 0 and 1 (default is 0.0). We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision


        .. code:: python

           > pairs = matcher.pairs(data)
           > scores = matcher.scores(pairs, threshold=0.5)
           > links = matcher.one_to_one(scores)
           > print(list(links))
           [((1, 2), 0.790),
            ((4, 5), 0.720),
            ((10, 11), 0.899)]

        """
        if threshold:
            scores = scores[scores['score'] > threshold]

        logger.debug("matching done, begin clustering")

        yield from clustering.greedyMatching(scores)

    def many_to_one(self,
                    scores: numpy.ndarray,
                    threshold: float = 0.0) -> Links:
        """
        From the similarity scores of pairs of records, decide which
        pairs refer to the same entity.

        Every record in data_1 can match at most one record from
        data_2, but more than one record from data_1 can match to the same
        record in data_2. See
        https://en.wikipedia.org/wiki/Surjective_function

        This method is good for when data_2 is a lookup table and data_1
        is messy, such as geocoding or matching against golden records.

        Yields pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

            threshold: Number between 0 and 1 (default is 0.0). We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

        .. code:: python

           > pairs = matcher.pairs(data)
           > scores = matcher.scores(pairs, threshold=0.5)
           > links = matcher.many_to_one(scores)
           > print(list(links))
           [((1, 2), 0.790),
            ((4, 5), 0.720),
            ((7, 2), 0.623),
            ((10, 11), 0.899)]

        """

        logger.debug("matching done, begin clustering")

        yield from clustering.pair_gazette_matching(scores, threshold, 1)


class GazetteerMatching(Matching):

    def __init__(self, num_cores: Optional[int], **kwargs) -> None:

        super().__init__(num_cores, **kwargs)

        self.temp_dir = tempfile.TemporaryDirectory()

        self.db = self.temp_dir.name + '/blocks.db'

        self.indexed_data: Dict[RecordID, RecordDict] = {}

    def _close(self):
        self.temp_dir.cleanup()

    def __del__(self):
        self._close()

    def index(self, data: Data) -> None:  # pragma: no cover
        """
        Add records to the index of records to match against. If a record in
        `canonical_data` has the same key as a previously indexed record, the
        old record will be replaced.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        self.fingerprinter.index_all(data)

        id_type = core.sqlite_id_type(data)

        con = sqlite3.connect(self.db)

        # Set journal mode to WAL.
        con.execute('pragma journal_mode=wal')

        con.execute('''CREATE TABLE IF NOT EXISTS indexed_records
                       (block_key text,
                        record_id {id_type},
                        UNIQUE(block_key, record_id))
                    '''.format(id_type=id_type))

        con.executemany("REPLACE INTO indexed_records VALUES (?, ?)",
                        self.fingerprinter(data.items(), target=True))

        con.execute('''CREATE INDEX IF NOT EXISTS
                       indexed_records_block_key_idx
                       ON indexed_records
                       (block_key)''')

        con.commit()
        con.close()

        self.indexed_data.update(data)

    def unindex(self, data: Data) -> None:  # pragma: no cover
        """
        Remove records from the index of records to match against.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        for field in self.fingerprinter.index_fields:
            self.fingerprinter.unindex({record[field]
                                        for record
                                        in data.values()},
                                       field)

        con = sqlite3.connect(self.db)
        con.executemany('''DELETE FROM indexed_records
                           WHERE record_id = ?''',
                        ((k, ) for k in data.keys()))

        con.commit()
        con.close()

        for k in data:
            del self.indexed_data[k]

    def blocks(self, data: Data) -> Blocks:
        """
        Yield groups of pairs of records that share fingerprints.

        Each group contains one record from data_1 paired with the records
        from the indexed records that data_1 shares a fingerprint with.

        Each pair within and among blocks will occur at most once. If
        you override this method, you need to take care to ensure that
        this remains true, as downstream methods, particularly
        :func:`many_to_n`, assumes that every pair of records is compared no
        more than once.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

        .. code:: python

            > pairs = matcher.pairs(data)
            > print(list(pairs))
            [[((1, {'name' : 'Pat', 'address' : '123 Main'}),
               (8, {'name' : 'Pat', 'address' : '123 Main'})),
              ((1, {'name' : 'Pat', 'address' : '123 Main'}),
               (9, {'name' : 'Sam', 'address' : '123 Main'}))
              ],
             [((2, {'name' : 'Sam', 'address' : '2600 State'}),
               (5, {'name' : 'Pam', 'address' : '2600 Stat'})),
              ((2, {'name' : 'Sam', 'address' : '123 State'}),
               (7, {'name' : 'Sammy', 'address' : '123 Main'}))
             ]]
        """

        id_type = core.sqlite_id_type(data)

        con = sqlite3.connect(self.db, check_same_thread=False)

        con.execute('BEGIN')

        con.execute('''CREATE TEMPORARY TABLE blocking_map
                       (block_key text, record_id {id_type})
                    '''.format(id_type=id_type))
        con.executemany("INSERT INTO blocking_map VALUES (?, ?)",
                        self.fingerprinter(data.items()))

        pairs = con.execute('''SELECT DISTINCT a.record_id, b.record_id
                               FROM blocking_map a
                               INNER JOIN indexed_records b
                               USING (block_key)
                               ORDER BY a.record_id''')

        pair_blocks = itertools.groupby(pairs,
                                        lambda x: x[0])

        for _, pair_block in pair_blocks:

            yield [((a_record_id, data[a_record_id]),
                    (b_record_id, self.indexed_data[b_record_id]))
                   for a_record_id, b_record_id
                   in pair_block]

        pairs.close()
        con.execute("ROLLBACK")
        con.close()

    def score(self,
              blocks: Blocks) -> Generator[numpy.ndarray, None, None]:
        """
        Scores groups of pairs of records. Yields structured numpy arrays
        representing pairs of records in the group and the associated
        probability that the pair is a match.

        Args:
            blocks: Iterator of blocks of records

        """

        matches = core.scoreGazette(blocks,
                                    self.data_model,
                                    self.classifier,
                                    self.num_cores)

        return matches

    def many_to_n(self,
                  score_blocks: Iterable[numpy.ndarray],
                  threshold: float = 0.0,
                  n_matches: int = 1) -> Links:
        """
        For each group of scored pairs, yield the highest scoring N pairs

        Args:
            score_blocks: Iterator of numpy `structured arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_,
                          each with a dtype of `[('pairs', id_type, 2),
                          ('score', 'f4')]` where dtype is either a str
                          or int, and score is a number between 0 and
                          1. The 'pairs' column contains pairs of ids of
                          the records compared and the 'score' column
                          should contains the similarity score for that
                          pair of records.

            threshold: Number between 0 and 1 (default is 0.0). We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

            n_matches: How many top scoring pairs to select per group

        """

        yield from clustering.gazetteMatching(score_blocks,
                                              threshold,
                                              n_matches)

    def search(self,
               data: Data,
               threshold: float = 0.0,
               n_matches: int = 1,
               generator: bool = False) -> LookupResults:  # pragma: no cover
        """
        Identifies pairs of records that could refer to the same entity,
        returns tuples containing tuples of possible matches, with a
        confidence score for each match. The record_ids within each
        tuple should refer to potential matches from a messy data
        record to canonical records. The confidence score is the
        estimated probability that the records refer to the same
        entity.

        Args:

            data: a dictionary of records from a messy
                  dataset, where the keys are record_ids and
                  the values are dictionaries with the keys
                  being field names.

            threshold: a number between 0 and 1 (default is
                       0.5). We will consider records as
                       potential duplicates if the predicted
                       probability of being a duplicate is
                       above the threshold.

                       Lowering the number will increase
                       recall, raising it will increase
                       precision
            n_matches: the maximum number of possible matches from
                       canonical_data to return for each record in
                       data. If set to `None` all possible
                       matches above the threshold will be
                       returned. Defaults to 1
            generator: when `True`, match will generate a sequence of
                       possible matches, instead of a list. Defaults
                       to `False` This makes `match` a lazy method.

        .. code:: python

            > matches = gazetteer.search(data, threshold=0.5, n_matches=2)
            > print(matches)
            [(((1, 6), 0.72),
              ((1, 8), 0.6)),
             (((2, 7), 0.72),),
             (((3, 6), 0.72),
              ((3, 8), 0.65)),
             (((4, 6), 0.96),
              ((4, 5), 0.63))]

        """
        blocks = self.blocks(data)
        pair_scores = self.score(blocks)
        search_results = self.many_to_n(pair_scores, threshold, n_matches)

        results = self._format_search_results(data, search_results)

        if generator:
            return results
        else:
            return list(results)

    def _format_search_results(self,
                               search_d: Data,
                               results: Links) -> LookupResults:

        seen: Set[RecordID] = set()

        for result in results:
            a: Optional[RecordID] = None
            b: RecordID
            score: float
            prepared_result: List[Tuple[RecordID, float]] = []
            for (a, b), score in result:  # type: ignore
                prepared_result.append((b, score))

            assert a is not None

            yield a, tuple(prepared_result)
            seen.add(a)

        for k in (search_d.keys() - seen):
            yield k, ()


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file.
    """

    def __init__(self,
                 settings_file: BinaryIO,
                 num_cores: Optional[int] = None,
                 **kwargs) -> None:  # pragma: no cover
        """
        Args:
            settings_file: A file object containing settings
                           info produced from the
                           :func:`~dedupe.api.ActiveMatching.write_settings` method.
            num_cores: the number of cpus to use for parallel
                       processing, defaults to the number of cpus
                       available on the machine. If set to 0, then
                       multiprocessing will be disabled.

        .. warning::

            If using multiprocessing on Windows or Mac OS X, then
            you must protect calls to the Dedupe methods with a
            `if __name__ == '__main__'` in your main module, see
            https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods

        """
        super().__init__(num_cores, **kwargs)

        try:
            self.data_model = pickle.load(settings_file)
            self.classifier = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")

        logger.info('Predicate set:')
        for predicate in self.predicates:
            logger.info(predicate)

        self._fingerprinter = blocking.Fingerprinter(self.predicates)


class ActiveMatching(Matching):
    """
    Class for training a matcher.
    """
    classifier = rlr.RegularizedLogisticRegression()

    def __init__(self,
                 variable_definition: Sequence[Mapping],
                 num_cores: Optional[int] = None,
                 **kwargs) -> None:
        """
        Args:
            variable_definition: A list of dictionaries describing
                                 the variables will be used for
                                 training a model. See :ref:`variable_definitions`

            num_cores: the number of cpus to use for parallel
                       processing. If set to `None`, uses all cpus
                       available on the machine. If set to 0, then
                       multiprocessing will be disabled.

        .. warning::

            If using multiprocessing on Windows or Mac OS X, then
            you must protect calls to the Dedupe methods with a
            `if __name__ == '__main__'` in your main module, see
            https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods

        """
        super().__init__(num_cores, **kwargs)

        self.data_model = datamodel.DataModel(variable_definition)

        self.training_pairs: TrainingData
        self.training_pairs = {'distinct': [],
                               'match': []}
        self.active_learner: Optional[Union[labeler.DedupeDisagreementLearner,
                                            labeler.RecordLinkDisagreementLearner]]
        self.active_learner = None

    def cleanup_training(self) -> None:  # pragma: no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_pairs
        del self.active_learner

    def _read_training(self, training_file: TextIO) -> None:
        '''
        Read training from previously built training data file object

        Args:
            training_file: file object containing the training data
        '''

        logger.info('reading training from file')
        training_pairs = json.load(training_file,
                                   object_hook=serializer._from_json)

        try:
            self.mark_pairs(training_pairs)
        except AttributeError as e:
            if "Attempting to fingerprint with an index predicate without indexing records" in str(e):
                raise UserWarning('Training data has records not known '
                                  'to the active learner. Read training '
                                  'in before initializing the active '
                                  'learner with the sample method, or '
                                  'use the prepare_training method.')
            else:
                raise

    def train(self,
              recall: float = 1.00,
              index_predicates: bool = True) -> None:  # pragma: no cover
        """
        Learn final pairwise classifier and fingerprinting rules. Requires that
        adequate training data has been already been provided.

        Args:
            recall: The proportion of true dupe pairs in our
                    training data that that the learned fingerprinting
                    rules must cover. If we lower the recall, there will
                    be pairs of true dupes that we will never
                    directly compare.

                    recall should be a float between 0.0 and 1.0.

            index_predicates: Should dedupe consider predicates
                              that rely upon indexing the
                              data. Index predicates can be slower
                              and take substantial memory.

        """
        assert self.active_learner is not None, \
               "Please initialize with the sample method"

        examples, y = flatten_training(self.training_pairs)
        self.classifier.fit(self.data_model.distances(examples), y)

        self.predicates = self.active_learner.learn_predicates(
            recall, index_predicates)
        self._fingerprinter = blocking.Fingerprinter(self.predicates)
        self.fingerprinter.reset_indices()

    def write_training(self, file_obj: TextIO) -> None:  # pragma: no cover
        """
        Write a JSON file that contains labeled examples

        :param file_obj: file object to write training data to

        .. code:: python

            with open('training.json', 'w') as f:
                matcher.write_training(f)

        """

        json.dump(self.training_pairs,
                  file_obj,
                  cls=serializer.TupleEncoder,
                  ensure_ascii=True)

    def write_settings(self,
                       file_obj: BinaryIO) -> None:  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        :param file_obj: file object to write settings data into

        .. code:: python

           with open('learned_settings', 'wb') as f:
               matcher.write_settings(f)

        """

        pickle.dump(self.data_model, file_obj)
        pickle.dump(self.classifier, file_obj)
        pickle.dump(self.predicates, file_obj)

    def uncertain_pairs(self) -> List[TrainingExample]:
        '''
        Returns a list of pairs of records from the sample of record pairs
        tuples that Dedupe is most curious to have labeled.

        This method is mainly useful for building a user interface for training
        a matching model.

       .. code:: python

          > pair = matcher.uncertain_pairs()
          > print(pair)
          [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]

        '''
        assert self.active_learner is not None, \
               "Please initialize with the sample method"
        return [self.active_learner.pop()]

    def mark_pairs(self, labeled_pairs: TrainingData) -> None:
        '''
        Add users labeled pairs of records to training data and update the
        matching model

        This method is useful for building a user interface for training a
        matching model or for adding training data from an existing source.

        Args:
            labeled_pairs: A dictionary with two keys, `match` and `distinct`
                           the values are lists that can contain pairs of
                           records

        .. code:: python

            labeled_examples = {'match'    : [],
                                'distinct' : [({'name' : 'Georgie Porgie'},
                                               {'name' : 'Georgette Porgette'})]
                                }
            matcher.mark_pairs(labeled_examples)

        '''
        self._checkTrainingPairs(labeled_pairs)

        self.training_pairs['match'].extend(labeled_pairs['match'])
        self.training_pairs['distinct'].extend(labeled_pairs['distinct'])

        if self.active_learner:
            examples, y = flatten_training(labeled_pairs)
            self.active_learner.mark(examples, y)

    def _checkTrainingPairs(self, labeled_pairs: TrainingData) -> None:
        try:
            labeled_pairs.items()
            labeled_pairs['match']
            labeled_pairs['distinct']
        except (AttributeError, KeyError):
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs['match']:
            pair = labeled_pairs['match'][0]
            self._checkRecordPair(pair)

        if labeled_pairs['distinct']:
            pair = labeled_pairs['distinct'][0]
            self._checkRecordPair(pair)

        if not labeled_pairs['distinct'] and not labeled_pairs['match']:
            warnings.warn("Didn't return any labeled record pairs")

    def _checkRecordPair(self, record_pair: TrainingExample) -> None:
        try:
            a, b = record_pair
        except ValueError:
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs")
        try:
            record_pair[0].keys() and record_pair[1].keys()
        except AttributeError:
            raise ValueError("A pair of record_pairs must be made up of two "
                             "dictionaries ")

        self.data_model.check(record_pair[0])
        self.data_model.check(record_pair[1])


class StaticDedupe(StaticMatching, DedupeMatching):
    """
    Class for deduplication using saved settings. If you have already
    trained a :class:`Dedupe` object and saved the settings, you can
    load the saved settings with StaticDedupe.

    """


class Dedupe(ActiveMatching, DedupeMatching):
    """
    Class for active learning deduplication. Use deduplication when you have
    data that can contain multiple records that can all refer to the same
    entity.
    """

    canopies = True
    ActiveLearner = labeler.DedupeDisagreementLearner

    def prepare_training(self,
                         data: Data,
                         training_file: TextIO = None,
                         sample_size: int = 1500,
                         blocked_proportion: float = 0.9,
                         original_length: int = None) -> None:
        '''
        Initialize the active learner with your data and, optionally,
        existing training data.

        Sets up the learner.

        Args:
            data: Dictionary of records, where the keys are
                  record_ids and the values are dictionaries
                  with the keys being field names
            training_file: file object containing training data
            sample_size: Size of the sample to draw
            blocked_proportion: The proportion of record pairs to be sampled from similar records, as opposed to randomly selected pairs. Defaults to 0.9.
            original_length: If `data` is a subsample of all your data,
                             `original_length` should be the size of
                             your complete data. By default,
                             `original_length` defaults to the length of
                             `data`.

        .. code:: python

           matcher.prepare_training(data_d, 150000, .5)

           # or
           with open('training_file.json') as f:
               matcher.prepare_training(data_d, training_file=f)


        '''

        if training_file:
            self._read_training(training_file)
        self._sample(data, sample_size, blocked_proportion, original_length)

    def _sample(self,
                data: Data,
                sample_size: int = 15000,
                blocked_proportion: float = 0.5,
                original_length: int = None) -> None:
        '''Draw a sample of record pairs from the dataset
        (a mix of random pairs & pairs of similar records)
        and initialize active learning with this sample


        :param data: Dictionary of records, where the keys are
                     record_ids and the values are dictionaries
                     with the keys being field names

        :param sample_size: Size of the sample to draw

        :param blocked_proportion: Proportion of the sample that will be blocked

        :param original_length: Length of original data, should be set
                                if `data` is a sample of full data

        '''
        self._checkData(data)

        if not original_length:
            original_length = len(data)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionary
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length,
                                                 index_include=examples)

        self.active_learner.mark(examples, y)

    def _checkData(self, data: Data) -> None:
        if len(data) == 0:
            raise ValueError(
                'Dictionary of records is empty.')

        self.data_model.check(next(iter(data.values())))


class Link(ActiveMatching):
    """
    Mixin Class for Active Learning Record Linkage
    """

    canopies = False
    ActiveLearner = labeler.RecordLinkDisagreementLearner

    def prepare_training(self,
                         data_1: Data,
                         data_2: Data,
                         training_file: Optional[TextIO] = None,
                         sample_size: int = 15000,
                         blocked_proportion: float = 0.5,
                         original_length_1: Optional[int] = None,
                         original_length_2: Optional[int] = None) -> None:
        '''
        Initialize the active learner with your data and, optionally,
        existing training data.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names
            data_2: Dictionary of records from second dataset, same
                    form as data_1
            training_file: file object containing training data

            sample_size: The size of the sample to draw. Defaults to 150,000

            blocked_proportion: The proportion of record pairs to
                                be sampled from similar records,
                                as opposed to randomly selected
                                pairs. Defaults to 0.5.
            original_length_1: If `data_1` is a subsample of your first dataset,
                               `original_length_1` should be the size of
                               the complete first dataset. By default,
                               `original_length_1` defaults to the length of
                               `data_1`
            original_length_2: If `data_2` is a subsample of your first dataset,
                               `original_length_2` should be the size of
                               the complete first dataset. By default,
                               `original_length_2` defaults to the length of
                               `data_2`

        .. code:: python

           matcher.prepare_training(data_1, data_2, 150000)

           with open('training_file.json') as f:
               matcher.prepare_training(data_1, data_2, training_file=f)

        '''

        if training_file:
            self._read_training(training_file)
        self._sample(data_1,
                     data_2,
                     sample_size,
                     blocked_proportion,
                     original_length_1,
                     original_length_2)

    def _sample(self,
                data_1: Data,
                data_2: Data,
                sample_size: int = 15000,
                blocked_proportion: float = 0.5,
                original_length_1: int = None,
                original_length_2: int = None) -> None:
        '''
        Draws a random sample of combinations of records from
        the first and second datasets, and initializes active
        learning with this sample

        :param data_1: Dictionary of records from first dataset, where the
                       keys are record_ids and the values are dictionaries
                       with the keys being field names
        :param data_2: Dictionary of records from second dataset, same
                       form as data_1
        :param sample_size: Size of the sample to draw
        '''
        self._checkData(data_1, data_2)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionaries
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data_1,
                                                 data_2,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length_1,
                                                 original_length_2,
                                                 index_include=examples)

        self.active_learner.mark(examples, y)

    def _checkData(self, data_1: Data, data_2: Data) -> None:
        if len(data_1) == 0:
            raise ValueError(
                'Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError(
                'Dictionary of records from second dataset is empty.')

        self.data_model.check(next(iter(data_1.values())))
        self.data_model.check(next(iter(data_2.values())))


class RecordLink(Link, RecordLinkMatching):
    """
    Class for active learning record linkage.

    Use RecordLinkMatching when you have two datasets that you want to
    join.
    """


class StaticRecordLink(StaticMatching, RecordLinkMatching):
    """
    Class for record linkage using saved settings. If you have already
    trained a RecordLink instance, you can load the saved settings with
    StaticRecordLink.
    """


class Gazetteer(Link, GazetteerMatching):
    """
    Class for active learning gazetteer matching.

    Gazetteer matching is for matching a messy data set against a
    'canonical dataset'.  This class is useful for such tasks as matching messy
    addresses against a clean list
    """


class StaticGazetteer(StaticMatching, GazetteerMatching):
    """
    Class for gazetter matching using saved settings.

    If you have already trained a :class:`Gazetteer` instance, you can
    load the saved settings with StaticGazetteer.
    """


class EmptyTrainingException(Exception):
    pass


class SettingsFileLoadingException(Exception):
    pass


def flatten_training(training_pairs: TrainingData) -> Tuple[List[TrainingExample], numpy.ndarray]:
    examples: List[TrainingExample] = []
    y = []

    for label in ('match', 'distinct'):
        label = cast(Literal['match', 'distinct'], label)

        pairs = training_pairs[label]
        examples.extend(pairs)
        encoded_y = 1 if label == 'match' else 0
        y.extend([encoded_y] * len(pairs))

    return examples, numpy.array(y)
