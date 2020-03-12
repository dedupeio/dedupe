#!/usr/bin/python
# -*- coding: utf-8 -*-
# -*- coding: future_fstrings -*-

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
import tempfile
import sqlite3
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

    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """

    def __init__(self, num_cores: Optional[int], **kwargs) -> None:
        print("Initializing Matching class")
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self._fingerprinter: Optional[blocking.Fingerprinter] = None
        self.data_model: datamodel.DataModel
        self.classifier: Classifier
        self.predicates: Sequence[dedupe.predicates.Predicate]
        self.loaded_indices = False

    @property
    def fingerprinter(self) -> blocking.Fingerprinter:
        if self._fingerprinter is None:
            raise ValueError('the record fingerprinter is not intialized, '
                             'please run the train method')
        return self._fingerprinter

    def thresholdBlocks(self, blocks, recall_weight=1.5):  # pragma: nocover
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
        candidate_records = itertools.chain.from_iterable(self._blockedPairs(blocks))

        probability = core.scoreDuplicates(candidate_records,
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

    def matchBlocks(self, blocks, classifier_threshold=.5,
                    cluster_threshold=0.5, *args, **kwargs):
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
        print("Matching.matchBlocks")
        candidate_records = itertools.chain.from_iterable(self._blockedPairs(blocks))
        matches = core.scoreDuplicates(candidate_records,
                                       self.data_model,
                                       self.classifier,
                                       self.num_cores,
                                       classifier_threshold)
        #print(matches)
        logger.debug("matching done, begin clustering")

        for cluster in self._cluster(matches, cluster_threshold, *args, **kwargs):
            yield cluster

        try:
            match_file = matches.filename
            del matches
            os.remove(match_file)
        except AttributeError:
            pass

    def write_settings(self, file_obj, index=False):  # pragma: no cover
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

    def _writeIndices(self, file_obj):
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

    def score(self, pairs, classifier_threshold=0.5):
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
                                           self.num_cores,
                                           classifier_threshold)
        except RuntimeError:
            raise RuntimeError('''
                You need to either turn off multiprocessing or protect
                the calls to the Dedupe methods with a
                `if __name__ == '__main__'` in your main module, see
                https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods''')

        return matches


class IntegralMatching(Matching):
    """
    This class is for linking class where we need to score all possible
    pairs before deciding on any matches
    """

    def score(self,
              pairs: RecordPairs) -> numpy.ndarray:
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

    def __init__(self, *args, **kwargs):
        print("Initializing DedupeMatching, calling super class Matching constructor")
        super().__init__(*args, **kwargs)
        self._cluster = clustering.cluster

    def partition(self, data, classifier_threshold=0.5, cluster_threshold=0.5,
                  generator=False):  # pragma: no cover
        """Identifies records that all refer to the same entity, returns
        tuples containing a set of record ids and a confidence score as a
        float between 0 and 1. The record_ids within each set should
        refer to the same entity and the confidence score is a measure
        of our confidence that all the records in a cluster refer to
        the same entity.

        This method should only used for small to moderately sized
        datasets for larger data, use matchBlocks

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

        Returns:
            clusters: (list)[tuple] list of clustered duplicates
                'idx' = id of record
                'scorex' = score, float between 0 and 1
                [
                    (('id1', 'id2', 'id3'), [score1, score2, score3])
                ]
        .. code:: python
           > clusters = matcher.partition(data, threshold=0.5)
           > print(duplicates)
           [((1, 2, 3), (0.790, 0.860, 0.790)),
            ((4, 5), (0.720, 0.720)),
            ((10, 11), (0.899, 0.899))]

        """

        blocked_pairs = self._blockData(data)
        print("DedupeMatching.partition")
        clusters = self.matchBlocks(blocked_pairs, classifier_threshold, cluster_threshold)
        if generator:
            return clusters
        else:
            return list(clusters)

    def threshold(self, data, recall_weight=1.5):  # pragma: no cover
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

        blocked_pairs = self._blockData(data)
        return self.thresholdBlocks(blocked_pairs, recall_weight)

    def _blockedPairs(self, blocks):
        """
        Generate tuples of pairs of records from a block of records

        Arguments:

        blocks -- an iterable sequence of blocked records
        """

        block, blocks = core.peek(blocks)
        self._checkBlock(block)

        combinations = itertools.combinations

        pairs = (combinations(sorted(block), 2) for block in blocks)

        return pairs

    def _blockData(self, data_d):
        """Use blocking rules from predicates to create blocks (preliminary clusters).

        The blocking.Fingerprinter.__call__ function takes the list of
        records (data_d) and creates groups of records based on
        the predicates computed during training. This function takes
        those results and converts them into a dictionary.

        blocks: (dict)
            key = (str) the stringified predicate rule
                A predicate rule can involve 1+ predicates. For example, a
                predicate rule might be:
                (   SimplePredicate: (wholeFieldPredicate, gender),
                    SimplePredicate: (wholeFieldPredicate, city)
                )
                For a record with city = 'Prince George' and gender = 'female',
                the stringified predicate rule would be:
                    'female:Prince George:0'
                The final number indicates the predicate rule number (0)
            value = (list)[str] list of record ids which are in this
                blocked cluster
        """

        blocks = {}
        coverage = {}

        if not self.loaded_indices:
            self.blocker.index_all(data_d)

        block_groups = itertools.groupby(self.blocker(data_d.items()),
                                         lambda x: x[1])
        print("DedupeMatching._blockData")

        for record_id, block in block_groups:
            print(f"Record id: {record_id}")
            block_keys = [block_key for block_key, _ in block]
            print(f"Block keys api: {block_keys}")
            coverage[record_id] = block_keys
            for block_key in block_keys:
                if block_key in blocks:
                    blocks[block_key].append(record_id)
                else:
                    blocks[block_key] = [record_id]

        if not self.loaded_indices:
            self.blocker.reset_indices()

        blocks = {block_key: record_ids for block_key, record_ids
                  in blocks.items() if len(record_ids) > 1}
        print("Blocks")
        print(blocks)
        coverage = {record_id: [k for k in cover if k in blocks]
                    for record_id, cover in coverage.items()}
        print(coverage)
        for block_key, block in blocks.items():
            processed_block = []
            print("New block")
            for record_id in block:
                smaller_blocks = {k for k in coverage[record_id]
                                  if k < block_key}
                processed_block.append((record_id, data_d[record_id], smaller_blocks))
                print(record_id)
            yield processed_block

    def _checkBlock(self, block):
        if block:
            try:
                id, record, smaller_ids = block[0]
            except (ValueError, KeyError):
                raise ValueError(
                    "Each item in a block must be a sequence of "
                    "record_id, record, and smaller ids and the "
                    "records also must be dictionaries")
            try:
                record.items()
                smaller_ids.isdisjoint([])
            except AttributeError:
                raise ValueError("The record must be a dictionary and "
                                 "smaller_ids must be a set")

            self.data_model.check(record)

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
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
                with a dtype of `[('pairs', id_type, 2),
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching

    def match(self, data_1, data_2, threshold=0.5, generator=False):  # pragma: no cover
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

        blocked_pairs = self._blockData(data_1, data_2)
        clusters = self.matchBlocks(blocked_pairs, threshold)

        if generator:
            return clusters
        else:
            return list(clusters)

    def threshold(self, data_1, data_2, recall_weight=1.5):  # pragma: no cover
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

        blocked_pairs = self._blockData(data_1, data_2)
        return self.thresholdBlocks(blocked_pairs, recall_weight)

    def _blockedPairs(self, blocks):
        """
        Generate tuples of pairs of records from a block of records

        Arguments:

        blocks -- an iterable sequence of blocked records
        """

        block, blocks = core.peek(blocks)
        self._checkBlock(block)

        product = itertools.product

        pairs = (product(base, target) for base, target in blocks)

        return pairs

    def _blockGenerator(self, messy_data, blocked_records):
        block_groups = itertools.groupby(self.blocker(messy_data.items()),
                                         lambda x: x[1])

        for i, (record_id, block_keys) in enumerate(block_groups):
            if i % 100 == 0:
                logger.info("%s records" % i)

            A = [(record_id, messy_data[record_id], set())]

            B = {}

            for block_key, _ in block_keys:
                if block_key in blocked_records:
                    B.update(blocked_records[block_key])

            B = [(rec_id, record, set())
                 for rec_id, record
                 in B.items()]

            if B:
                yield (A, B)

    def _blockData(self, data_1, data_2):

        blocked_records = {}

        if not self.loaded_indices:
            self.blocker.index_all(data_2)

        for block_key, record_id in self.blocker(data_2.items(), target=True):
            block = blocked_records.setdefault(block_key, {})
            block[record_id] = data_2[record_id]

        for each in self._blockGenerator(data_1, blocked_records):
            yield each

    def _checkBlock(self, block):
        if block:
            try:
                base, target = block
            except ValueError:
                raise ValueError("Each block must be a made up of two "
                                 "sequences, (base_sequence, target_sequence)")

            if base:
                try:
                    base_id, base_record, base_smaller_ids = base[0]
                except ValueError:
                    raise ValueError(
                        "Each sequence must be made up of 3-tuple "
                        "like (record_id, record, covered_blocks)")
                self.data_model.check(base_record)
            if target:
                try:
                    target_id, target_record, target_smaller_ids = target[0]
                except ValueError:
                    raise ValueError(
                        "Each sequence must be made up of 3-tuple "
                        "like (record_id, record, covered_blocks)")
                self.data_model.check(target_record)


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file,
    extends Matching.

    Public methods:
    - __init__
    """

    def __init__(self,
                 settings_file,
                 num_cores=None, **kwargs):  # pragma: no cover
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
        file see the method [`write_settings`][[api.py#write_settings]].
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

        self.blocker = blocking.Fingerprinter(self.predicates)

    def _loadIndices(self, settings_file):
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
    ActiveLearner = None

    """
    Class for training dedupe extends Matching.

    Public methods:
    - __init__
    - readTraining
    - train
    - write_settings
    - write_training
    - uncertainPairs
    - markPairs
    - cleanupTraining
    """

    def __init__(self,
                 variable_definition,
                 data_sample=None,
                 num_cores=None, **kwargs):
        """
        Initialize from a data model and data sample.

        #### Example usage

            # initialize from a defined set of fields
            variable_definition = [{'field' : 'Site name', 'type': 'String'},
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
        print("Initializing ActiveMatching class")
        Matching.__init__(self, num_cores, **kwargs)

        self.data_model = datamodel.DataModel(variable_definition)

        if data_sample is not None:
            raise UserWarning(
                'data_sample is deprecated, use the .sample method')

        self.active_learner = None

        self.training_pairs = OrderedDict({u'distinct': [],
                                           u'match': []})

        self.blocker = None

    def cleanupTraining(self):  # pragma: no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_pairs
        del self.active_learner

    def _read_training(self, training_file):
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

    def train(self, recall=0.95, index_predicates=True):  # pragma: no cover
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
        print("ActiveMatching.train")
        examples, y = flatten_training(self.training_pairs)
        print("Fit classifier with active label training data")
        self.classifier.fit(self.data_model.distances(examples), y)
        print(f"Number of matches: {len(self.training_pairs['match'])}")
        self.predicates = self.active_learner.learn_predicates(
            recall, index_predicates)
        self.blocker = blocking.Fingerprinter(self.predicates)
        self.blocker.reset_indices()

    def write_training(self, file_obj):  # pragma: no cover
        """
        Write to a json file that contains labeled examples

        Keyword arguments:
        file_obj -- file object to write training data to
        """

        json.dump(self.training_pairs,
                  file_obj,
                  default=serializer._to_json,
                  ensure_ascii=True)

    def uncertainPairs(self):
        '''
        Provides a list of the pairs of records that dedupe is most
        curious to learn if they are matches or distinct.

        Useful for user labeling.

        '''

        return self.active_learner.pop()

    def markPairs(self, labeled_pairs):
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

    def _checkTrainingPairs(self, labeled_pairs):
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

    def _checkRecordPair(self, record_pair):
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
    Class for active learning deduplication. Use deduplication when you have
    data that can contain multiple records that can all refer to the same
    entity.
    """

    canopies = True

    def prepare_training(self,
                         data: Data,
                         training_file: TextIO = None,
                         sample_size: int = 1500,
                         blocked_proportion: float = 0.9,
                         original_length: int = None) -> None:
        """Sets up the learner.

        Initialize the active learner with your data and, optionally,
        existing training data.

        Args:
            data: (dict) Dictionary of records, where the keys are
                  record_ids and the values are dictionaries
                  with the keys being field names
            training_file: file object containing training data
            sample_size: (int) Size of the sample to draw
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

        """

        print("Preparing training")
        if training_file:
            print("Reading active labels from training file")
            self._read_training(training_file)
        self._sample(data, sample_size, blocked_proportion, original_length)

    def _sample(self,
                data: Data,
                sample_size: int = 15000,
                blocked_proportion: float = 0.5,
                original_length: int = None) -> None:
        """Draw a sample of record pairs from the dataset
        (a mix of random pairs & pairs of similar records)
        and initialize active learning with this sample

        Args:

            data: (dict) Dictionary of records, where the keys are
                record_ids and the values are dictionaries with the keys being
                field names
            sample_size: (int) Size of the sample to draw
            blocked_proportion: Proportion of the sample that will be blocked
            original_length: Length of original data, should be set if `data` is
                                   a sample of full data


        """
        print("api.Dedupe.sample")
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
        print("Marking active training data")
        self.active_learner.mark(examples, y)

    def _checkData(self, data):
        """Check that data is not empty and that each record has required fields.
        """
        print("Checking input data is valid")
        if len(data) == 0:
            raise ValueError(
                'Dictionary of records is empty.')

        self.data_model.check(next(iter(data.values())))


class StaticRecordLink(RecordLinkMatching, StaticMatching):
    """
    Mixin Class for Static Record Linkage
    """


class RecordLink(RecordLinkMatching, ActiveMatching):
    """
    Mixin Class for Active Learning Record Linkage

    Public Methods
    - sample
    """
    canopies = False

    def prepare_training(self,
                         data_1,
                         data_2,
                         training_file=None,
                         sample_size=15000,
                         blocked_proportion=0.5,
                         original_length_1=None,
                         original_length_2=None):
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
            self._read_training(training_file)
        self.sample(data_1,
                    data_2,
                    sample_size,
                    blocked_proportion,
                    original_length_1,
                    original_length_2)

    def sample(self,
               data_1,
               data_2,
               sample_size=15000,
               blocked_proportion=0.5,
               original_length_1=None,
               original_length_2=None):
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

    def _checkData(self, data_1, data_2):
        if len(data_1) == 0:
            raise ValueError(
                'Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError(
                'Dictionary of records from second dataset is empty.')

        self.data_model.check(next(iter(data_1.values())))
        self.data_model.check(next(iter(data_2.values())))


class GazetteerMatching(RecordLinkMatching):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching

    def _blockData(self, messy_data):
        for each in self._blockGenerator(messy_data, self.blocked_records):
            yield each

    def index(self, data):  # pragma: no cover

        self.blocker.index_all(data)

        for block_key, record_id in self.blocker(data.items(), target=True):
            if block_key not in self.blocked_records:
                self.blocked_records[block_key] = {}
            self.blocked_records[block_key][record_id] = data[record_id]

    def unindex(self, data):  # pragma: no cover

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

    def matchBlocks(self, blocks, threshold=.5, *args, **kwargs):
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
        blocked_pairs = self._blockData(messy_data)

        clusters = self.matchBlocks(blocked_pairs, threshold, n_matches)

        clusters = (cluster for cluster in clusters if len(cluster))

        if generator:
            return clusters
        else:
            return list(clusters)

    def write_settings(self, file_obj, index=False):  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        Keyword arguments:
        file_obj -- file object to write settings data into
        """
        super().write_settings(file_obj, index)

        if index:
            pickle.dump(self.blocked_records, file_obj)


class Gazetteer(RecordLink, GazetteerMatching):

    def __init__(self, *args, **kwargs):  # pragma: no cover
        super().__init__(*args, **kwargs)
        self.blocked_records = OrderedDict({})


class StaticGazetteer(StaticRecordLink, GazetteerMatching):

    def __init__(self, *args, **kwargs):
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


def flatten_training(training_pairs):
    """Convert active label training pairs to two lists.

    Args:
        :training_pairs: (dict) dictionary of either 'match' or 'distinct', where
            'match' is a list of pairs of records which are the same, and
            'distinct' is a list of pairs of records which are different

        {
            'match': [
                [record_1, record_2]
            ],
            'distinct': [
                [record_1, record_3]
            ]
        }

    Returns:
        :examples: (list)[list] ordered list of all the record pairs (distinct and match)

            [
                [record_1, record_2],
                [record_1, record_3]
            ]
        :y: (list)[int] list of either 1 or 0, corresponding to examples list
            1 = match
            0 = distinct
    """
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
