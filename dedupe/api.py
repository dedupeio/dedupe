#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""
from __future__ import print_function, division
from future.utils import viewitems, viewvalues
from builtins import super

import itertools
import logging
import pickle
import multiprocessing
import warnings
import os
from collections import OrderedDict

import numpy
import simplejson as json
import rlr

import dedupe.core as core
import dedupe.serializer as serializer
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.datamodel as datamodel
import dedupe.labeler as labeler

logger = logging.getLogger(__name__)


class Matching(object):
    """
    Base Class for Record Matching Classes

    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """

    def __init__(self, num_cores):
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self.loaded_indices = False

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

    def matchBlocks(self, blocks, threshold=.5, *args, **kwargs):
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
        candidate_records = itertools.chain.from_iterable(self._blockedPairs(blocks))

        matches = core.scoreDuplicates(candidate_records,
                                       self.data_model,
                                       self.classifier,
                                       self.num_cores,
                                       threshold=0)

        logger.debug("matching done, begin clustering")

        for cluster in self._cluster(matches, threshold, *args, **kwargs):
            yield cluster

        try:
            match_file = matches.filename
            del matches
            os.remove(match_file)
        except AttributeError:
            pass

    def writeSettings(self, file_obj, index=False):  # pragma: no cover
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
        super().__init__(*args, **kwargs)
        self._cluster = clustering.cluster

    def match(self, data, threshold=0.5, generator=False):  # pragma: no cover
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
        blocked_pairs = self._blockData(data)
        clusters = self.matchBlocks(blocked_pairs, threshold)
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

        blocks = {}
        coverage = {}

        if not self.loaded_indices:
            self.blocker.indexAll(data_d)

        block_groups = itertools.groupby(self.blocker(viewitems(data_d)),
                                         lambda x: x[1])

        for record_id, block in block_groups:
            block_keys = [block_key for block_key, _ in block]
            coverage[record_id] = block_keys
            for block_key in block_keys:
                if block_key in blocks:
                    blocks[block_key].append(record_id)
                else:
                    blocks[block_key] = [record_id]

        if not self.loaded_indices:
            self.blocker.resetIndices()

        blocks = {block_key: record_ids for block_key, record_ids
                  in blocks.items() if len(record_ids) > 1}

        coverage = {record_id: [k for k in cover if k in blocks]
                    for record_id, cover in coverage.items()}

        for block_key, block in blocks.items():
            processed_block = []
            for record_id in block:
                smaller_blocks = {k for k in coverage[record_id]
                                  if k < block_key}
                processed_block.append((record_id, data_d[record_id], smaller_blocks))

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
        block_groups = itertools.groupby(self.blocker(viewitems(messy_data)),
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
            self.blocker.indexAll(data_2)

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
    - writeSettings
    - writeTraining
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

    def readTraining(self, training_file):
        '''
        Read training from previously built training data file object

        Arguments:

        training_file -- file object containing the training data
        '''

        logger.info('reading training from file')
        training_pairs = json.load(training_file,
                                   cls=serializer.dedupe_decoder)
        self.markPairs(training_pairs)

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
                            be slower and take susbstantial memory.

                            Defaults to True.
        """
        examples, y = flatten_training(self.training_pairs)
        self.classifier.fit(self.data_model.distances(examples), y)

        self.predicates = self.active_learner.learn_predicates(
            recall, index_predicates)
        self.blocker = blocking.Blocker(self.predicates)
        self.blocker.resetIndices()

    def writeTraining(self, file_obj):  # pragma: no cover
        """
        Write to a json file that contains labeled examples

        Keyword arguments:
        file_obj -- file object to write training data to
        """

        json.dump(self.training_pairs,
                  file_obj,
                  default=serializer._to_json,
                  tuple_as_array=False,
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
    Mixin Class for Active Learning Deduplication

    Public Methods
    - sample
    """
    canopies = True

    def sample(self, data, sample_size=15000,
               blocked_proportion=0.5, original_length=None):
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

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length)

    def _checkData(self, data):
        if len(data) == 0:
            raise ValueError(
                'Dictionary of records is empty.')

        self.data_model.check(next(iter(viewvalues(data))))


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

    def sample(self, data_1, data_2, sample_size=15000,
               blocked_proportion=.5, original_length_1=None,
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

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data_1,
                                                 data_2,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length_1,
                                                 original_length_2)

    def _checkData(self, data_1, data_2):
        if len(data_1) == 0:
            raise ValueError(
                'Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError(
                'Dictionary of records from second dataset is empty.')

        self.data_model.check(next(iter(viewvalues(data_1))))
        self.data_model.check(next(iter(viewvalues(data_2))))


class GazetteerMatching(RecordLinkMatching):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching

    def _blockData(self, messy_data):
        for each in self._blockGenerator(messy_data, self.blocked_records):
            yield each

    def index(self, data):  # pragma: no cover

        self.blocker.indexAll(data)

        for block_key, record_id in self.blocker(data.items(), target=True):
            if block_key not in self.blocked_records:
                self.blocked_records[block_key] = {}
            self.blocked_records[block_key][record_id] = data[record_id]

    def unindex(self, data):  # pragma: no cover

        for field in self.blocker.index_fields:
            self.blocker.unindex((record[field]
                                  for record
                                  in viewvalues(data)),
                                 field)

        for block_key, record_id in self.blocker(viewitems(data)):
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

        blocked_pairs = self._blockData(messy_data)
        return self.thresholdBlocks(blocked_pairs, recall_weight)

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
