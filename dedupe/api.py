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
from collections import OrderedDict
import abc

import numpy
import json
import rlr

import dedupe.core as core
import dedupe.serializer as serializer
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.datamodel as datamodel
import dedupe.labeler as labeler

logger = logging.getLogger(__name__)

from typing import Iterator, Tuple, Mapping, Sequence, Union, Optional, Any, Set, Type, Iterable, Generator, cast, Dict, List, BinaryIO, TextIO, overload
IndicesIterator = Iterator[Tuple[int, int]]
RecordID = Union[int, str]
Record = Tuple[RecordID, Mapping[str, Any]]
RecordPair = Tuple[Record, Record]
RecordPairs = Iterator[RecordPair]
_Queue = Union[multiprocessing.dummy.Queue, multiprocessing.Queue]
_SimpleQueue = Union[multiprocessing.dummy.Queue, multiprocessing.SimpleQueue]
Cluster = Tuple[RecordID]
Clusters = Iterable[Cluster]
Data = Mapping[RecordID, Mapping[str, Any]]
Block = Tuple[Sequence[Record], Sequence[Record]]
Blocks = Iterator[Block]
TrainingExample = Tuple[Mapping[str, Any], Mapping[str, Any]]
TrainingData = Mapping[str, List[TrainingExample]]


class Matching(object):
    """
    Base Class for Record Matching Classes

    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """

    def __init__(self, num_cores: Optional[int], **kwargs):
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self.loaded_indices = False

    def threshold_pairs(self, pairs, recall_weight: float=1.5) -> float:  # pragma: nocover
        """
        Returns the threshold that maximizes the expected F score, a
        weighted average of precision and recall for a sample of
        blocked data.

        Arguments:

        blocks -- Sequence of tuples of records, where each tuple is a
                  set of records covered by a blocking predicate

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.

        """
        probability = core.scoreDuplicates(pairs,
                                           self.data_model,
                                           self.classifier,
                                           self.num_cores)['score']

        probability = probability.copy()
        probability.sort()
        probability = probability[::-1]

        expected_dupes = numpy.cumsum(probability)

        recall = expected_dupes / expected_dupes[-1]
        precision = expected_dupes / numpy.arange(1, len(expected_dupes) + 1)

        score = recall * precision / (recall + recall_weight ** 2 * precision)

        i = numpy.argmax(score)

        logger.info('Maximum expected recall and precision')
        logger.info('recall: %2.3f', recall[i])
        logger.info('precision: %2.3f', precision[i])
        logger.info('With threshold: %2.3f', probability[i])

        return probability[i]

    def score(self, pairs, *args, **kwargs):
        """
        Partitions blocked data and generates a sequence of clusters,
        where each cluster is a tuple of record ids

        Keyword arguments:

        blocks -- Sequence of tuples of records, where each tuple is a
                  set of records covered by a blocking predicate

        threshold -- Number between 0 and 1 (default is .5). We will
                      only consider as duplicates record pairs as
                      duplicates if their estimated duplicate
                      likelihood is greater than the threshold.

                      Lowering the number will increase recall,
                      raising it will increase precision

        """

        matches = core.scoreDuplicates(pairs,
                                       self.data_model,
                                       self.classifier,
                                       self.num_cores,
                                       threshold=0)

        return matches

    def cluster(self, matches, threshold, *args, **kwargs):

        logger.debug("matching done, begin clustering")

        yield from self._cluster(matches, threshold, *args, **kwargs)

    def writeSettings(self, file_obj: BinaryIO, index: bool=False) -> None:  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        Keyword arguments:
        file_obj -- file object to write settings data into
        """

        pickle.dump(self.data_model, file_obj)
        pickle.dump(self.classifier, file_obj)
        pickle.dump(self.predicates, file_obj)

        if index:
            self._writeIndices(file_obj)

    def _writeIndices(self, file_obj: BinaryIO) -> None:
        indices = {}
        doc_to_ids = {}
        canopies = {}
        for full_predicate in self.predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index') and predicate.index:
                    doc_to_ids[predicate] = dict(predicate.index._doc_to_id)
                    if hasattr(predicate, "canopy"):
                        canopies[predicate] = predicate.canopy
                    else:
                        try:
                            indices[predicate] = predicate.index._index
                        except AttributeError:
                            pass

        pickle.dump(canopies, file_obj)
        pickle.dump(indices, file_obj)
        pickle.dump(doc_to_ids, file_obj)


class DedupeMatching(Matching):
    """
    Class for Deduplication, extends Matching.

    Use DedupeMatching when you have a dataset that can contain
    multiple references to the same entity.

    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """

    ActiveLearner = labeler.DedupeDisagreementLearner

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cluster = clustering.cluster

    def match(self, data: Data, threshold: float=0.5, generator: bool=False) -> Clusters:  # pragma: no cover
        """Identifies records that all refer to the same entity, returns
        tuples

        containing a set of record ids and a confidence score as a
        float between 0 and 1. The record_ids within each set should
        refer to the same entity and the confidence score is a measure
        of our confidence that all the records in a cluster refer to
        the same entity.

        This method should only used for small to moderately sized
        datasets for larger data, use matchBlocks

        Arguments:

        data -- Dictionary of records, where the keys are record_ids
                and the values are dictionaries with the keys being
                field names

        threshold -- Number between 0 and 1 (default is .5). We will
                      consider records as potential duplicates if the
                      predicted probability of being a duplicate is
                      above the threshold.

                      Lowering the number will increase recall,
                      raising it will increase precision

        """
        pairs = self.pairs(data)
        pair_scores = self.score(pairs)
        clusters = self.cluster(pair_scores, threshold)

        try:
            if generator:
                return clusters
            else:
                clusters = list(clusters)
                return clusters
        finally:
            try:
                mmap_file = pair_scores.filename
                del pair_scores
                os.remove(mmap_file)
            except AttributeError:
                pass

    def threshold(self, data: Data, recall_weight: float=1.5) -> float:  # pragma: no cover
        """
        Returns the threshold that maximizes the expected F score,
        a weighted average of precision and recall for a sample of
        data.

        Arguments:
        data          -- Dictionary of records, where the keys are record_ids
                         and the values are dictionaries with the keys being
                         field names

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.
        """
        
        pairs = self.pairs(data)
        return self.threshold_pairs(pairs, recall_weight)

    def pairs(self, data_d: Data) -> RecordPairs:

        if not self.loaded_indices:
            self.blocker.indexAll(data_d)

        blocked_records: Dict[str, List[RecordID]] = {}
        for block_key, record_id in self.blocker(data_d.items()):
            blocked_records.setdefault(block_key, []).append(record_id)

        block_groups = itertools.groupby(self.blocker(data_d.items()),
                                         lambda x: x[1])

        product = itertools.product

        for a_record_id, block_keys in block_groups:

            A: List[Record] = [(a_record_id, data_d[a_record_id])]

            b_record_ids: set = set()
            for block_key, _ in block_keys:
                if block_key in blocked_records:
                    b_record_ids.update(blocked_records[block_key])

            B: List[Record] = [(b_record_id, data_d[b_record_id])
                               for b_record_id
                               in b_record_ids
                               if b_record_id > a_record_id]

            if B:
                yield from product(A, B)

        if not self.loaded_indices:
            self.blocker.resetIndices()

class RecordLinkMatching(Matching):
    """
    Class for Record Linkage, extends Matching.

    Use RecordLinkMatching when you have two datasets that you want to merge
    where each dataset, individually, contains no duplicates.

    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """

    ActiveLearner = labeler.RecordLinkDisagreementLearner

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching

    def match(self, data_1: Data, data_2: Data, threshold: float=0.5, generator: bool=False) -> Clusters:  # pragma: no cover
        """
        Identifies pairs of records that refer to the same entity, returns
        tuples containing a set of record ids and a confidence score as a float
        between 0 and 1. The record_ids within each set should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        This method should only used for small to moderately sized datasets
        for larger data, use matchBlocks

        Arguments:
        data_1    -- Dictionary of records from first dataset, where the
                     keys are record_ids and the values are dictionaries
                     with the keys being field names

        data_2    -- Dictionary of records from second dataset, same form
                     as data_1

        threshold -- Number between 0 and 1 (default is .5). We will consider
                     records as potential duplicates if the predicted
                     probability of being a duplicate is above the threshold.

                     Lowering the number will increase recall, raising it
                     will increase precision
        """
        pairs = self.pairs(data_1, data_2)
        pair_scores = self.score(pairs)
        clusters = self.cluster(pair_scores, threshold)

        try:
            if generator:
                return clusters
            else:
                clusters = list(clusters)
                return clusters
        finally:
            try:
                mmap_file = pair_scores.filename
                del pair_scores
                os.remove(mmap_file)
            except AttributeError:
                pass

    def threshold(self, data_1: Data, data_2: Data, recall_weight: float=1.5) -> float:  # pragma: no cover
        """
        Returns the threshold that maximizes the expected F score,
        a weighted average of precision and recall for a sample of
        data.

        Arguments:
        data_1        --  Dictionary of records from first dataset, where the
                          keys are record_ids and the values are dictionaries
                          with the keys being field names

        data_2        --  Dictionary of records from second dataset, same form
                          as data_1

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.
        """

        blocks = self._blockData(data_1, data_2)
        return self.thresholdBlocks(blocks, recall_weight)

    def pairs(self, data_1: Data, data_2: Data) -> RecordPairs:

        if not self.loaded_indices:
            self.blocker.indexAll(data_2)

        blocked_records: Dict[str, List] = {}

        for block_key, record_id in self.blocker(data_2.items(), target=True):
            blocked_records.setdefault(block_key, []).append(record_id)

        block_groups = itertools.groupby(self.blocker(data_1.items()),
                                         lambda x: x[1])

        product = itertools.product

        for i, (a_record_id, block_keys) in enumerate(block_groups):
            if i % 100 == 0:
                logger.info("%s records" % i)

            A: List[Record] = [(a_record_id, data_1[a_record_id])]

            b_record_ids: set = set()
            for block_key, _ in block_keys:
                if block_key in blocked_records:
                    b_record_ids.update(blocked_records[block_key])

            B: List[Record] = [(rec_id, data_2[rec_id])
                               for rec_id
                               in b_record_ids]

            if B:
                yield from product(A, B)

class GazetteerMatching(Matching):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching

    def _blockData(self, messy_data: Data) -> Blocks:
        for each in self._blockGenerator(messy_data, self.blocked_records):
            yield each

    def index(self, data: Data):  # pragma: no cover

        self.blocker.indexAll(data)

        for block_key, record_id in self.blocker(data.items(), target=True):
            if block_key not in self.blocked_records:
                self.blocked_records[block_key] = {}
            self.blocked_records[block_key][record_id] = data[record_id]

    def unindex(self, data: Data):  # pragma: no cover

        for field in self.blocker.index_fields:
            self.blocker.unindex((record[field]
                                  for record
                                  in data.values()),
                                 field)

        for block_key, record_id in self.blocker(data.items()):
            try:
                del self.blocked_records[block_key][record_id]
            except KeyError:
                pass

    def matchBlocks(self, blocks: Blocks, threshold: float=.5, *args, **kwargs) -> Clusters:
        """
        Partitions blocked data and generates a sequence of clusters, where
        each cluster is a tuple of record ids

        Keyword arguments:

        blocks -- Sequence of tuples of records, where each tuple is a
                  set of records covered by a blocking predicate

        threshold -- Number between 0 and 1 (default is .5). We will
                      only consider as duplicates record pairs as
                      duplicates if their estimated duplicate
                      likelihood is greater than the threshold.

                      Lowering the number will increase recall,
                      raising it will increase precision

        """
        candidate_records = self._blockedPairs(blocks)

        matches = core.scoreGazette(candidate_records,
                                    self.data_model,
                                    self.classifier,
                                    self.num_cores,
                                    threshold=threshold)

        logger.debug("matching done, begin clustering")

        return self._cluster(matches, *args, **kwargs)

    def match(self, messy_data, threshold=0.5, n_matches=1, generator=False):  # pragma: no cover
        """Identifies pairs of records that refer to the same entity, returns
        tuples containing a set of record ids and a confidence score as a float
        between 0 and 1. The record_ids within each set should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        This method should only used for small to moderately sized datasets
        for larger data, use matchBlocks

        Arguments:
        messy_data -- Dictionary of records from messy dataset, where the
                      keys are record_ids and the values are dictionaries with
                      the keys being field names

        threshold -- Number between 0 and 1 (default is .5). We will consider
                     records as potential duplicates if the predicted
                     probability of being a duplicate is above the threshold.

                     Lowering the number will increase recall, raising it
                     will increase precision

        n_matches -- Maximum number of possible matches from the canonical
                     record set to match against each record in the messy
                     record set
        """
        blocks = self._blockData(messy_data)

        clusters = self.matchBlocks(blocks, threshold, n_matches)

        clusters = (cluster for cluster in clusters if len(cluster))

        if generator:
            return clusters
        else:
            return list(clusters)

    def threshold(self, messy_data, recall_weight=1.5):  # pragma: no cover
        """
        Returns the threshold that maximizes the expected F score,
        a weighted average of precision and recall for a sample of
        data.

        Arguments:
        messy_data -- Dictionary of records from messy dataset, where the
                      keys are record_ids and the values are dictionaries with
                      the keys being field names

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.
        """

        blocks = self._blockData(messy_data)
        return self.thresholdBlocks(blocks, recall_weight)

    def writeSettings(self, file_obj, index=False):  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        Keyword arguments:
        file_obj -- file object to write settings data into
        """
        super().writeSettings(file_obj, index)

        if index:
            pickle.dump(self.blocked_records, file_obj)


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file,
    extends Matching.

    Public methods:
    - __init__
    """

    def __init__(self,
                 settings_file: BinaryIO,
                 num_cores: int=None, **kwargs) -> None:  # pragma: no cover
        """
        Initialize from a settings file
        #### Example usage

            # initialize from a settings file
            with open('my_learned_settings', 'rb') as f:
                deduper = dedupe.StaticDedupe(f)

        #### Keyword arguments

        `settings_file`
        A file object containing settings data.


        Settings files are typically generated by saving the settings
        learned from ActiveMatching. If you need details for this
        file see the method [`writeSettings`][[api.py#writesettings]].
        """
        Matching.__init__(self, num_cores, **kwargs)

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

        try:
            self._loadIndices(settings_file)
        except EOFError:
            pass
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")

        logger.info(self.predicates)

        self.blocker = blocking.Blocker(self.predicates)

    def _loadIndices(self, settings_file: BinaryIO) -> None:
        canopies = pickle.load(settings_file)
        indices = pickle.load(settings_file)
        doc_to_ids = pickle.load(settings_file)

        for full_predicate in self.predicates:
            for predicate in full_predicate:
                if hasattr(predicate, "index") and predicate.index is None:
                    predicate.index = predicate.initIndex()
                    max_id = max(doc_to_ids[predicate].values())
                    predicate.index._doc_to_id = core.Enumerator(max_id + 1,
                                                                 doc_to_ids[predicate])

                    if hasattr(predicate, "canopy"):
                        predicate.canopy = canopies[predicate]
                    else:
                        try:
                            predicate.index._index = indices[predicate]
                        except KeyError:
                            pass

        self.loaded_indices = True


class ActiveMatching(Matching):
    classifier = rlr.RegularizedLogisticRegression()

    """
    Class for training dedupe extends Matching.

    Public methods:
    - __init__
    - readTraining
    - train
    - writeSettings
    - writeTraining
    - uncertainPairs
    - markPairs
    - cleanupTraining
    """

    def __init__(self,
                 variable_definition: Mapping,
                 num_cores: int=None, **kwargs) -> None:
        """
        Initialize from a data model and data sample.

        #### Example usage

            # initialize from a defined set of fields
            fields = [{'field' : 'Site name', 'type': 'String'},
                      {'field' : 'Address', 'type': 'String'},
                      {'field' : 'Zip', 'type': 'String',
                       'Has Missing':True},
                      {'field' : 'Phone', 'type': 'String',
                       'Has Missing':True},
                     ]

            deduper = dedupe.Dedupe(fields)


        #### Additional detail

        A field definition is a list of dictionaries where each dictionary
        describes a variable to use for comparing records.

        For details about variable types, check the documentation.
        <https://docs.dedupe.io/>`_
        """
        Matching.__init__(self, num_cores, **kwargs)

        self.data_model = datamodel.DataModel(variable_definition)

        self.training_pairs: TrainingData
        self.training_pairs = OrderedDict({u'distinct': [],
                                           u'match': []})


    def cleanupTraining(self) -> None:  # pragma: no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_pairs
        del self.active_learner

    def readTraining(self, training_file: TextIO) -> None:
        '''
        Read training from previously built training data file object

        Arguments:

        training_file -- file object containing the training data
        '''

        logger.info('reading training from file')
        training_pairs = json.load(training_file,
                                   cls=serializer.dedupe_decoder)

        try:
            self.markPairs(training_pairs)
        except AttributeError as e:
            if "Attempting to block with an index predicate without indexing records" in str(e):
                raise UserWarning('Training data has records not known '
                                  'to the active learner. Read training '
                                  'in before initializing the active '
                                  'learner with the sample method, or '
                                  'use the prepare_training method.')
            else:
                raise

    def train(self, recall: float=0.95, index_predicates: bool=True) -> None:  # pragma: no cover
        """
        Keyword arguments:

        maximum_comparisons -- The maximum number of comparisons a
                               blocking rule is allowed to make.

                               Defaults to 1000000

        recall -- The proportion of true dupe pairs in our training
                  data that that we the learned blocks must cover. If
                  we lower the recall, there will be pairs of true
                  dupes that we will never directly compare.

                  recall should be a float between 0.0 and 1.0, the default
                  is 0.95

        index_predicates -- Should dedupe consider predicates that
                            rely upon indexing the data. Index predicates can
                            be slower and take substantial memory.

                            Defaults to True.
        """
        examples, y = flatten_training(self.training_pairs)
        self.classifier.fit(self.data_model.distances(examples), y)

        self.predicates = self.active_learner.learn_predicates(
            recall, index_predicates)
        self.blocker = blocking.Blocker(self.predicates)
        self.blocker.resetIndices()

    def writeTraining(self, file_obj: TextIO) -> None:  # pragma: no cover
        """
        Write to a json file that contains labeled examples

        Keyword arguments:
        file_obj -- file object to write training data to
        """

        json.dump(self.training_pairs,
                  file_obj,
                  default=serializer._to_json,
                  ensure_ascii=True)

    def uncertainPairs(self) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        '''
        Provides a list of the pairs of records that dedupe is most
        curious to learn if they are matches or distinct.

        Useful for user labeling.

        '''

        return self.active_learner.pop()

    def markPairs(self, labeled_pairs: TrainingData) -> None:
        '''
        Argument :

        labeled_pairs -- A dictionary with two keys, `match` and `distinct`
                         the values are lists that can contain pairs of records
        '''
        self._checkTrainingPairs(labeled_pairs)

        for label, examples in labeled_pairs.items():
            self.training_pairs[label].extend(examples)

        if self.active_learner:
            examples, y = flatten_training(labeled_pairs)
            self.active_learner.mark(examples, y)

    def _checkTrainingPairs(self, labeled_pairs: TrainingData) -> None:
        try:
            labeled_pairs.items()
            labeled_pairs[u'match']
            labeled_pairs[u'distinct']
        except (AttributeError, KeyError):
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs[u'match']:
            pair = labeled_pairs[u'match'][0]
            self._checkRecordPair(pair)

        if labeled_pairs[u'distinct']:
            pair = labeled_pairs[u'distinct'][0]
            self._checkRecordPair(pair)

        if not labeled_pairs[u'distinct'] and not labeled_pairs[u'match']:
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


class StaticDedupe(DedupeMatching, StaticMatching):
    """
    Mixin Class for Static Deduplication
    """


class Dedupe(DedupeMatching, ActiveMatching):
    """
    Mixin Class for Active Learning Deduplication

    Public Methods
    - sample
    """
    canopies = True

    def prepare_training(self,
                         data: Data,
                         training_file: TextIO=None,
                         sample_size: int=1500,
                         blocked_proportion: float=0.9,
                         original_length: int=None) -> None:
        '''
        Sets up the learner.
        Arguments:

        Arguments: data -- Dictionary of records, where the keys are
                           record_ids and the values are dictionaries
                           with the keys being field names
        training_file -- file object containing training data
        sample_size         -- Size of the sample to draw
        blocked_proportion  -- Proportion of the sample that will be blocked
        original_length     -- Length of original data, should be set if
                               `data` is a sample of full data

        '''

        if training_file:
            self.readTraining(training_file)
        self.sample(data, sample_size, blocked_proportion, original_length)

    def sample(self, data: Data, sample_size: int=15000,
               blocked_proportion: float=0.5, original_length: int=None) -> None:
        '''Draw a sample of record pairs from the dataset
        (a mix of random pairs & pairs of similar records)
        and initialize active learning with this sample

        Arguments: data -- Dictionary of records, where the keys are
        record_ids and the values are dictionaries with the keys being
        field names
        sample_size         -- Size of the sample to draw
        blocked_proportion  -- Proportion of the sample that will be blocked
        original_length     -- Length of original data, should be set if `data` is
                               a sample of full data


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

    def _checkData(self, data: Data):
        if len(data) == 0:
            raise ValueError(
                'Dictionary of records is empty.')

        self.data_model.check(next(iter(data.values())))


class StaticRecordLink(RecordLinkMatching, StaticMatching):
    """
    Mixin Class for Static Record Linkage
    """


class Link(object):
    """
    Mixin Class for Active Learning Record Linkage

    Public Methods
    - sample
    - prepare_training
    """

    canopies = False

    def prepare_training(self,
                         data_1: Data,
                         data_2: Data,
                         training_file: Optional[TextIO]=None,
                         sample_size: int=15000,
                         blocked_proportion: float=0.5,
                         original_length_1: Optional[int]=None,
                         original_length_2: Optional[int]=None) -> None:
        '''
        Sets up the learner.
        Arguments:

        data_1      -- Dictionary of records from first dataset, where the
                       keys are record_ids and the values are dictionaries
                       with the keys being field names
        data_2      -- Dictionary of records from second dataset, same
                       form as data_1
        training_file -- file object containing training data
        '''

        if training_file:
            self.readTraining(training_file)
        self.sample(data_1,
                    data_2,
                    sample_size,
                    blocked_proportion,
                    original_length_1,
                    original_length_2)

    def sample(self,
               data_1: Data,
               data_2: Data,
               sample_size: int=15000,
               blocked_proportion: float=0.5,
               original_length_1: int=None,
               original_length_2: int=None) -> None:
        '''
        Draws a random sample of combinations of records from
        the first and second datasets, and initializes active
        learning with this sample

        Arguments:

        data_1      -- Dictionary of records from first dataset, where the
                       keys are record_ids and the values are dictionaries
                       with the keys being field names
        data_2      -- Dictionary of records from second dataset, same
                       form as data_1

        sample_size -- Size of the sample to draw
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

    
class RecordLink(Link, RecordLinkMatching, ActiveMatching):
    """
    Active Matching for Record Link
    """


class Gazetteer(GazetteerMatching, Link, ActiveMatching):

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        super().__init__(*args, **kwargs)
        self.blocked_records: Mapping = OrderedDict({})


class StaticGazetteer(GazetteerMatching, StaticMatching):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        settings_file = args[0]

        try:
            self.blocked_records = pickle.load(settings_file)
        except EOFError:
            self.blocked_records = OrderedDict({})
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")


class EmptyTrainingException(Exception):
    pass


class SettingsFileLoadingException(Exception):
    pass


def flatten_training(training_pairs: TrainingData):
    examples = []
    y = []
    for label, pairs in training_pairs.items():
        for pair in pairs:
            if label == 'match':
                y.append(1)
                examples.append(pair)
            elif label == 'distinct':
                y.append(0)
                examples.append(pair)

    return examples, numpy.array(y)
