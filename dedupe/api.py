#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""
from __future__ import print_function, division
from future.utils import viewitems, viewvalues, viewkeys

import itertools
import logging
import pickle
import multiprocessing
import random
import warnings
import copy
import os
from collections import defaultdict, OrderedDict

import numpy
import simplejson as json
import rlr

import dedupe.sampling as sampling
import dedupe.core as core
import dedupe.training as training
import dedupe.serializer as serializer
import dedupe.predicates as predicates
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.datamodel as datamodel

logger = logging.getLogger(__name__)


class Matching(object):
    """
    Base Class for Record Matching Classes

    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """

    def __init__(self):
        pass

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

        probability = core.scoreDuplicates(self._blockedPairs(blocks),
                                           self.data_model,
                                           self.classifier,
                                           self.num_cores)['score']

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
        Partitions blocked data and returns a list of clusters, where
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

        matches = core.scoreDuplicates(candidate_records,
                                       self.data_model,
                                       self.classifier,
                                       self.num_cores,
                                       threshold)

        logger.debug("matching done, begin clustering")

        clusters = self._cluster(matches, threshold, *args, **kwargs)

        try:
            match_file = matches.filename
            del matches
            os.remove(match_file)
        except AttributeError:
            pass

        return clusters

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
                        indices[predicate] = predicate.index._index

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

    def __init__(self, *args, **kwargs):
        super(DedupeMatching, self).__init__(*args, **kwargs)
        self._cluster = clustering.cluster
        self._linkage_type = "Dedupe"

    def match(self, data, threshold=0.5):  # pragma: no cover
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
        return self.matchBlocks(blocked_pairs, threshold)

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

        return itertools.chain.from_iterable(pairs)

    def _checkBlock(self, block):
        if block:
            try:
                if len(block[0]) < 3:
                    raise ValueError(
                        "Each item in a block must be a sequence "
                        "of record_id, record, and smaller ids and "
                        "the records also must be dictionaries")
            except:
                raise ValueError(
                        "Each item in a block must be a sequence of "
                        "record_id, record, and smaller ids and the "
                        "records also must be dictionaries")
            try:
                block[0][1].items()
                block[0][2].isdisjoint([])
            except:
                raise ValueError("The record must be a dictionary and "
                                 "smaller_ids must be a set")

            self.data_model.check(block[0][1])

    def _blockData(self, data_d):

        blocks = defaultdict(dict)

        if not self.loaded_indices:
            self.blocker.indexAll(data_d)

        for block_key, record_id in self.blocker(viewitems(data_d)):
            blocks[block_key][record_id] = data_d[record_id]

        seen_blocks = set()
        blocks = [records for records in viewvalues(blocks)
                  if len(records) > 1 and
                  not (frozenset(records.keys()) in seen_blocks or
                       seen_blocks.add(frozenset(records.keys())))]

        for block in self._redundantFree(blocks):
            yield block

    def _redundantFree(self, blocks):
        """
        Redundant-free Comparisons from Kolb et al, "Dedoop:
        Efficient Deduplication with Hadoop"
        http://dbs.uni-leipzig.de/file/Dedoop.pdf
        """
        coverage = defaultdict(list)

        for block_id, records in enumerate(blocks):

            for record_id, record in viewitems(records):
                coverage[record_id].append(block_id)

        for block_id, records in enumerate(blocks):
            if block_id % 10000 == 0:
                logger.info("%s blocks" % block_id)

            marked_records = []
            for record_id, record in viewitems(records):
                smaller_ids = {covered_id for covered_id
                               in coverage[record_id]
                               if covered_id < block_id}
                marked_records.append((record_id, record, smaller_ids))

            yield marked_records


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

    def __init__(self, *args, **kwargs):
        super(RecordLinkMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching
        self._linkage_type = "RecordLink"

    def match(self, data_1, data_2, threshold=0.5):  # pragma: no cover
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
        return self.matchBlocks(blocked_pairs, threshold)

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

        return itertools.chain.from_iterable(pairs)

    def _checkBlock(self, block):
        if block:
            try:
                base, target = block
            except:
                raise ValueError("Each block must be a made up of two "
                                 "sequences, (base_sequence, target_sequence)")

            if base:
                if len(base[0]) < 3:
                    raise ValueError(
                            "Each sequence must be made up of 3-tuple "
                            "like (record_id, record, covered_blocks)")
                self.data_model.check(base[0][1])
            if target:
                if len(target[0]) < 3:
                    raise ValueError(
                              "Each sequence must be made up of 3-tuple "
                              "like (record_id, record, covered_blocks)")
                self.data_model.check(target[0][1])

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

        blocked_records = defaultdict(dict)

        if not self.loaded_indices:
            self.blocker.indexAll(data_2)

        for block_key, record_id in self.blocker(data_2.items()):
            blocked_records[block_key][record_id] = data_2[record_id]

        for each in self._blockGenerator(data_1, blocked_records):
            yield each


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file,
    extends Matching.

    Public methods:
    - __init__

    """

    def __init__(self,
                 settings_file,
                 num_cores=None):  # pragma: no cover
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
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        try:
            self.data_model = pickle.load(settings_file)
            self.classifier = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")

        self.loaded_indices = False

        try:
            self._loadIndices(settings_file)
        except EOFError:
            pass
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:
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
                    predicate.index._doc_to_id = doc_to_ids[predicate]
                    if hasattr(predicate, "canopy"):
                        predicate.canopy = canopies[predicate]
                    else:
                        predicate.index._index = indices[predicate]

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
                 variable_definition,
                 data_sample=None,
                 num_cores=None):
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

            data_sample = [
                           (
                            (854, {'city': 'san francisco',
                             'address': '300 de haro st.',
                             'name': "sally's cafe & bakery",
                             'cuisine': 'american'}),
                            (855, {'city': 'san francisco',
                             'address': '1328 18th st.',
                             'name': 'san francisco bbq',
                             'cuisine': 'thai'})
                             )
                            ]



            deduper = dedupe.Dedupe(fields, data_sample)


        #### Additional detail

        A field definition is a list of dictionaries where each dictionary
        describes a variable to use for comparing records.

        For details about variable types, check the documentation.
        <http://dedupe.readthedocs.org>`_

        In the data_sample, each element is a tuple of two
        records. Each record is, in turn, a tuple of the record's key and
        a record dictionary.

        In in the record dictionary the keys are the names of the
        record field and values are the record values.
        """
        self.data_model = datamodel.DataModel(variable_definition)

        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        if data_sample:
            self._checkDataSample(data_sample)
            self.data_sample = data_sample
            self.activeLearner = training.ActiveLearning(self.data_sample,
                                                         self.data_model,
                                                         self.num_cores)
        else:
            self.data_sample = []
            self.activeLearner = None

        # Override _loadSampledRecords() to load blocking data from
        # data_sample.
        self._loadSampledRecords(data_sample)

        training_dtype = [('label', 'S8'),
                          ('distances', 'f4',
                           (len(self.data_model), ))]

        self.training_data = numpy.zeros(0, dtype=training_dtype)
        self.training_pairs = OrderedDict({u'distinct': [],
                                           u'match': []})

        self.blocker = None
        self.loaded_indices = False

    def cleanupTraining(self):  # pragma: no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_data
        del self.training_pairs
        del self.activeLearner
        del self.data_sample

    def readTraining(self, training_file):
        '''
        Read training from previously built training data file object

        Arguments:

        training_file -- file object containing the training data
        '''

        logger.info('reading training from file')

        training_pairs = json.load(training_file,
                                   cls=serializer.dedupe_decoder)

        if not any(training_pairs.values()):
            raise EmptyTrainingException(
                "The training file seems to contain no training examples")

        for (label, examples) in training_pairs.items():
            if examples:
                self._checkRecordPairType(examples[0])

            examples = core.freezeData(examples)

            training_pairs[label] = examples
            self.training_pairs[label].extend(examples)

        self._addTrainingData(training_pairs)

        self._trainClassifier()

    def train(self, ppc=None, uncovered_dupes=None, maximum_comparisons=1000000, recall=0.95, index_predicates=True):  # pragma: no cover
        """Keyword arguments:

        maximum_comparisons -- The maximum number of comparisons a
                               blocking rule is allowed to make.

                               Defaults to 1000000

        recall -- The proportion of true dupe pairs in our training
                  data that that we the learned blocks must cover. If
                  we lower the recall, there will be pairs of true
                  dupes that we will never directly compare.

                  recall should be a float between 0.0 and 1.0, the default
                  is 0.975

        index_predicates -- Should dedupe consider predicates that
                            rely upon indexing the data. Index predicates can
                            be slower and take susbstantial memory.

                            Defaults to True.
        """
        if ppc is not None:
            warnings.warn('`ppc` is a deprecated argument to train. '
                          'Use `maximum_comparisons` to set the maximum '
                          'number records a block is allowed to cover')

        if uncovered_dupes is not None:
            warnings.warn('`uncovered_dupes` is a deprecated argument '
                          'to train. Use recall to set the proportion '
                          'of true pairs that the blocking rules must cover')

        self._trainClassifier()
        self._trainBlocker(maximum_comparisons,
                           recall,
                           index_predicates)

    def _trainClassifier(self):  # pragma: no cover
        labels = numpy.array(self.training_data['label'] == b'match',
                             dtype='int8')
        examples = self.training_data['distances']

        self.classifier.fit(examples, labels)

    def _trainBlocker(self, maximum_comparisons, recall, index_predicates):  # pragma: no cover
        matches = self.training_pairs['match'][:]

        predicate_set = self.data_model.predicates(index_predicates,
                                                   self.canopies)

        block_learner = self._blockLearner(predicate_set)

        self.predicates = block_learner.learn(matches,
                                              maximum_comparisons,
                                              recall)

        self.blocker = blocking.Blocker(self.predicates)

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

        if self.training_data.shape[0] == 0:
            rand_int = random.randint(0, len(self.data_sample) - 1)
            random_pair = self.data_sample[rand_int]
            exact_match = (random_pair[0], random_pair[0])
            self._addTrainingData({u'match': [exact_match, exact_match],
                                   u'distinct': [random_pair]})

        self._trainClassifier()

        bias = len(self.training_pairs[u'match'])
        if bias:
            bias /= (bias +
                     len(self.training_pairs[u'distinct']))

        min_examples = min(len(self.training_pairs[u'match']),
                           len(self.training_pairs[u'distinct']))

        regularizer = 10

        bias = ((0.5 * min_examples + bias * regularizer) /
                (min_examples + regularizer))

        return self.activeLearner.uncertainPairs(self.classifier, bias)

    def markPairs(self, labeled_pairs):
        '''
        Add a labeled pairs of record to dedupes training set and update the
        matching model

        Argument :

        labeled_pairs -- A dictionary with two keys, `match` and `distinct`
                         the values are lists that can contain pairs of records

        '''
        try:
            labeled_pairs.items()
            labeled_pairs[u'match']
            labeled_pairs[u'distinct']
        except:
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs[u'match']:
            pair = labeled_pairs[u'match'][0]
            self._checkRecordPairType(pair)

        if labeled_pairs[u'distinct']:
            pair = labeled_pairs[u'distinct'][0]
            self._checkRecordPairType(pair)

        if not labeled_pairs[u'distinct'] and not labeled_pairs[u'match']:
            warnings.warn("Didn't return any labeled record pairs")

        for label, pairs in labeled_pairs.items():
            self.training_pairs[label].extend(core.freezeData(pairs))

        self._addTrainingData(labeled_pairs)

    def _checkRecordPairType(self, record_pair):
        try:
            record_pair[0]
        except:
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs (ordered sequences of length 2)")

        if len(record_pair) != 2:
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs")
        try:
            record_pair[0].keys() and record_pair[1].keys()
        except:
            raise ValueError("A pair of record_pairs must be made up of two "
                             "dictionaries ")

        self.data_model.check(record_pair[0])
        self.data_model.check(record_pair[1])

    def _checkDataSample(self, data_sample):
        try:
            len(data_sample)
        except TypeError:
            raise ValueError("data_sample must be a sequence")

        if len(data_sample):
            self._checkRecordPairType(data_sample[0])

        else:
            warnings.warn("You submitted an empty data_sample")

    def _addTrainingData(self, labeled_pairs):
        """
        Appends training data to the training data collection.
        """

        for label, examples in labeled_pairs.items():
            n_examples = len(examples)
            labels = [label] * n_examples

            new_data = numpy.empty(n_examples,
                                   dtype=self.training_data.dtype)

            new_data['label'] = labels
            new_data['distances'] = self.data_model.distances(examples)

            self.training_data = numpy.append(self.training_data,
                                              new_data)

    def _loadSample(self, data_sample):

        self._checkDataSample(data_sample)

        self.data_sample = data_sample

        self.activeLearner = training.ActiveLearning(self.data_sample,
                                                     self.data_model,
                                                     self.num_cores)

    def _loadSampledRecords(self, data_sample):
        """Override to load blocking data from data_sample."""


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
               blocked_proportion=0.5):
        '''Draw a sample of record pairs from the dataset
        (a mix of random pairs & pairs of similar records)
        and initialize active learning with this sample

        Arguments: data -- Dictionary of records, where the keys are
        record_ids and the values are dictionaries with the keys being
        field names

        sample_size         -- Size of the sample to draw
        blocked_proportion  -- Proportion of the sample that will be blocked
        '''
        data = core.index(data)
        self.sampled_records = Sample(data, 900)

        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(self.data_model.predicates(index_predicates=False))

        data = sampling.randomDeque(data)
        blocked_sample_keys = sampling.dedupeBlockedSample(blocked_sample_size,
                                                           predicates,
                                                           data)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = set(core.randomPairs(len(data),
                                                  random_sample_size))
        data = dict(data)

        data_sample = [(data[k1], data[k2])
                       for k1, k2
                       in blocked_sample_keys | random_sample_keys]

        data_sample = core.freezeData(data_sample)

        self._loadSample(data_sample)

    def _blockLearner(self, predicates):
        return training.DedupeBlockLearner(predicates,
                                           self.sampled_records)

    def _loadSampledRecords(self, data_sample):
        if data_sample:
            recs = itertools.chain.from_iterable(data_sample)
            data = dict(enumerate(recs))
            self.sampled_records = Sample(data, 900)
        else:
            self.sampled_records = None


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

    def sample(self, data_1, data_2, sample_size=150000,
               blocked_proportion=.5):
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
        if len(data_1) == 0:
            raise ValueError(
                'Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError(
                'Dictionary of records from second dataset is empty.')

        if len(data_1) > len(data_2):
            data_1, data_2 = data_2, data_1

        data_1 = core.index(data_1)
        self.sampled_records_1 = Sample(data_1, 500)

        offset = len(data_1)
        data_2 = core.index(data_2, offset)
        self.sampled_records_2 = Sample(data_2, 500)

        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(self.data_model.predicates(index_predicates=False))

        deque_1 = sampling.randomDeque(data_1)
        deque_2 = sampling.randomDeque(data_2)

        blocked_sample_keys = sampling.linkBlockedSample(blocked_sample_size,
                                                         predicates,
                                                         deque_1,
                                                         deque_2)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = core.randomPairsMatch(len(deque_1),
                                                   len(deque_2),
                                                   random_sample_size)

        random_sample_keys = {(a, b + offset)
                              for a, b in random_sample_keys}

        data_sample = ((data_1[k1], data_2[k2])
                       for k1, k2
                       in blocked_sample_keys | random_sample_keys)

        data_sample = core.freezeData(data_sample)

        self._loadSample(data_sample)

    def _blockLearner(self, predicates):
        return training.RecordLinkBlockLearner(predicates,
                                               self.sampled_records_1,
                                               self.sampled_records_2)

    def _loadSampledRecords(self, data_sample):
        if data_sample:
            data_1 = dict(enumerate(x for (x, y) in data_sample))
            offset = len(data_1)
            data_2 = dict(enumerate((y for (x, y) in data_sample), offset))
            self.sampled_records_1 = Sample(data_1, 500)
            self.sampled_records_2 = Sample(data_2, 500)
        else:
            self.sampled_records_1 = None
            self.sampled_records_2 = None


class GazetteerMatching(RecordLinkMatching):

    def __init__(self, *args, **kwargs):
        super(GazetteerMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching
        self._linkage_type = "GazetteerMatching"

    def _blockData(self, messy_data):
        for each in self._blockGenerator(messy_data, self.blocked_records):
            yield each

    def index(self, data):  # pragma: no cover

        self.blocker.indexAll(data)

        for block_key, record_id in self.blocker(data.items()):
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

    def match(self, messy_data, threshold=0.5, n_matches=1):  # pragma: no cover
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
        return self.matchBlocks(blocked_pairs, threshold, n_matches)

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
        super(GazetteerMatching, self).writeSettings(file_obj, index)

        if index:
            pickle.dump(self.blocked_records, file_obj)


class Gazetteer(RecordLink, GazetteerMatching):

    def __init__(self, *args, **kwargs):  # pragma: no cover
        super(Gazetteer, self).__init__(*args, **kwargs)
        self.blocked_records = OrderedDict({})


class StaticGazetteer(StaticRecordLink, GazetteerMatching):

    def __init__(self, *args, **kwargs):
        super(StaticGazetteer, self).__init__(*args, **kwargs)

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
        except:
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")


class EmptyTrainingException(Exception):
    pass


class SettingsFileLoadingException(Exception):
    pass


class Sample(dict):

    def __init__(self, d, sample_size):
        if len(d) <= sample_size:
            super(Sample, self).__init__(d)
        else:
            super(Sample, self).__init__({k: d[k]
                                          for k
                                          in random.sample(viewkeys(d),
                                                           sample_size)})
        self.original_length = len(d)
