#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""
from __future__ import print_function, division
from future.utils import viewitems, viewvalues

import itertools
import logging
import pickle
import numpy
import multiprocessing
import random
import warnings
import copy
import os
from collections import defaultdict, OrderedDict
import simplejson as json
import rlr

import dedupe
import dedupe.sampling as sampling
import dedupe.core as core
import dedupe.training as training
import dedupe.serializer as serializer
import dedupe.crossvalidation as crossvalidation
import dedupe.predicates as predicates
import dedupe.blocking as blocking
import dedupe.clustering as clustering
from dedupe.datamodel import DataModel

logger = logging.getLogger(__name__)


class Matching(object):
    """
    Base Class for Record Matching Classes
    
    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """
    def __init__(self) :
        pass

    def thresholdBlocks(self, blocks, recall_weight=1.5): # pragma : nocover
        """
        Returns the threshold that maximizes the expected F score,
        a weighted average of precision and recall for a sample of
        blocked data. 

        Keyword arguments:
        blocks --        Sequence of tuples of records, where each
                         tuple is a set of records covered by a blocking
                         predicate

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.
        """


        probability = core.scoreDuplicates(self._blockedPairs(blocks), 
                                           self.data_model, 
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

    def matchBlocks(self, blocks, threshold=.5, *args, **kwargs): # pragma : no cover
        """
        Partitions blocked data and returns a list of clusters, where
        each cluster is a tuple of record ids

        Keyword arguments:
        blocks --     Sequence of tuples of records, where each
                      tuple is a set of records covered by a blocking
                      predicate
                                          
        threshold --  Number between 0 and 1 (default is .5). We will
                      only consider as duplicates record pairs as
                      duplicates if their estimated duplicate likelihood is
                      greater than the threshold.

                      Lowering the number will increase recall, raising it
                      will increase precision
                              

        """
        # Setting the cluster threshold this ways is not principled,
        # but seems to reliably help performance
        cluster_threshold = threshold * 0.7

        candidate_records = self._blockedPairs(blocks)
        
        matches = core.scoreDuplicates(candidate_records,
                                       self.data_model,
                                       self.num_cores,
                                       threshold)

        logger.debug("matching done, begin clustering")

        clusters = self._cluster(matches, 
                                 cluster_threshold, *args, **kwargs)

        try :
            match_file = matches.filename
            del matches
            os.remove(match_file)
        except AttributeError :
            pass
        
        return clusters

    def _checkRecordType(self, record) :
        for field_comparator in self.data_model.field_comparators :
            field = field_comparator[0]
            if field not in record :
                raise ValueError("Records do not line up with data model. "
                                 "The field '%s' is in data_model but not "
                                 "in a record" % field)


    def _logLearnedWeights(self): # pragma: no cover
        """
        Log learned weights and bias terms
        """
        logger.info('Learned Weights')
        for (key_1, value_1) in self.data_model.items():
            try:
                for field in value_1 :
                    logger.info((field.name, field.weight))
            except TypeError :
                logger.info((key_1, value_1))


class DedupeMatching(Matching) :
    """
    Class for Deduplication, extends Matching.
    
    Use DedupeMatching when you have a dataset that can contain 
    multiple references to the same entity.
    
    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """
    
    def __init__(self, *args, **kwargs) :
        super(DedupeMatching, self).__init__(*args, **kwargs)
        self._cluster = clustering.cluster
        self._linkage_type = "Dedupe"

    def match(self, data, threshold = 0.5) : # pragma : no cover
        """
        Identifies records that all refer to the same entity, returns tuples
        containing a set of record ids and a confidence score as a float between 0
        and 1. The record_ids within each set should refer to the
        same entity and the confidence score is a measure of our confidence that
        all the records in a cluster refer to the same entity.
        
        This method should only used for small to moderately sized datasets
        for larger data, use matchBlocks
        
        Arguments:
        data      --  Dictionary of records, where the keys are record_ids
                      and the values are dictionaries with the keys being
                      field names
                                          
        threshold --  Number between 0 and 1 (default is .5). We will consider
                      records as potential duplicates if the predicted probability
                      of being a duplicate is above the threshold.

                      Lowering the number will increase recall, raising it
                      will increase precision
                             
        """
        blocked_pairs = self._blockData(data)
        return self.matchBlocks(blocked_pairs, threshold)

    def threshold(self, data, recall_weight = 1.5) : # pragma : no cover
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

    def _blockedPairs(self, blocks) :
        """
        Generate tuples of pairs of records from a block of records
        
        Arguments:
        
        blocks -- an iterable sequence of blocked records
        """
        
        block, blocks = core.peek(blocks)
        self._checkBlock(block)

        combinations = itertools.combinations

        pairs = (combinations(block, 2) for block in blocks)

        return itertools.chain.from_iterable(pairs) 

    def _checkBlock(self, block) :
        if block :
            try :
                if len(block[0]) < 3 :
                    raise ValueError("Each item in a block must be a "
                                     "sequence of record_id, record, and smaller ids "
                                     "and the records also must be dictionaries")
            except :
                raise ValueError("Each item in a block must be a "
                                 "sequence of record_id, record, and smaller ids "
                                 "and the records also must be dictionaries")
            try :
                block[0][1].items()
                block[0][2].isdisjoint([])
            except :
                raise ValueError("The record must be a dictionary and "
                                 "smaller_ids must be a set")

        
            self._checkRecordType(block[0][1])

    def _blockData(self, data_d):

        blocks = defaultdict(dict)

        for field in self.blocker.index_fields :
            unique_fields = {record[field]
                             for record 
                             in viewvalues(data_d)
                             if record[field]}

            self.blocker.index(unique_fields, field)

        for block_key, record_id in self.blocker(viewitems(data_d)) :
            blocks[block_key][record_id] = data_d[record_id]

        self.blocker.resetIndices()

        blocks = (records for records in blocks.values()
                  if len(records) > 1)
        
        blocks = {frozenset(d.keys()) : d for d in blocks}
        blocks = blocks.values()

        for block in self._redundantFree(blocks) :
            yield block

    def _redundantFree(self, blocks) :
        """
        Redundant-free Comparisons from Kolb et al, "Dedoop:
        Efficient Deduplication with Hadoop"
        http://dbs.uni-leipzig.de/file/Dedoop.pdf
        """
        coverage = defaultdict(list)

        for block_id, records in enumerate(blocks) :

            for record_id, record in viewitems(records) :
                coverage[record_id].append(block_id)

        for block_id, records in enumerate(blocks) :
            if block_id % 10000 == 0 :
                logger.info("%s blocks" % block_id)

            marked_records = []
            for record_id, record in viewitems(records) :
                smaller_ids = {covered_id for covered_id 
                               in coverage[record_id] 
                               if covered_id < block_id}
                marked_records.append((record_id, record, smaller_ids))

            yield marked_records


class RecordLinkMatching(Matching) :
    """
    Class for Record Linkage, extends Matching.
    
    Use RecordLinkMatching when you have two datasets that you want to merge
    where each dataset, individually, contains no duplicates.
    
    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """

    def __init__(self, *args, **kwargs) :
        super(RecordLinkMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching
        self._linkage_type = "RecordLink"

    def match(self, data_1, data_2, threshold = 0.5) : # pragma : no cover
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

    def threshold(self, data_1, data_2, recall_weight = 1.5) : # pragma : no cover
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
x        """

        blocked_pairs = self._blockData(data_1, data_2)
        return self.thresholdBlocks(blocked_pairs, recall_weight)

    def _blockedPairs(self, blocks) :
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
        
    def _checkBlock(self, block) :
        if block :
            try :
                base, target = block
            except :
                raise ValueError("Each block must be a made up of two "
                                 "sequences, (base_sequence, target_sequence)")

            if base :
                if len(base[0]) < 3 :
                    raise ValueError("Each sequence must be made up of 3-tuple "
                                     "like (record_id, record, covered_blocks)")
                self._checkRecordType(base[0][1])
            if target :
                if len(target[0]) < 3 :
                    raise ValueError("Each sequence must be made up of 3-tuple "
                                     "like (record_id, record, covered_blocks)")
                self._checkRecordType(target[0][1])

    def _blockGenerator(self, messy_data, blocked_records) :
        block_groups = itertools.groupby(self.blocker(viewitems(messy_data)), 
                                         lambda x : x[1])

        for i, (record_id, block_keys) in enumerate(block_groups) :
            if i % 100 == 0 :
                logger.info("%s records" % i)

            A = [(record_id, messy_data[record_id], set())]

            B = {}

            for block_key, _ in block_keys :
                if block_key in blocked_records :
                    B.update(blocked_records[block_key])

            B = [(rec_id, record, set())
                 for rec_id, record
                 in B.items()]

            if B :
                yield (A, B)


    def _blockData(self, data_1, data_2) :

        blocked_records = defaultdict(dict)

        for field in self.blocker.index_fields :
            fields_2 = (record[field]
                        for record 
                        in viewvalues(data_2))

            self.blocker.index(set(fields_2), field)

        for block_key, record_id in self.blocker(data_2.items()) :
            blocked_records[block_key][record_id] = data_2[record_id]

        for each in self._blockGenerator(data_1, blocked_records) :
            yield each

        self.blocker.resetIndices()


class StaticMatching(Matching) :
    """
    Class for initializing a dedupe object from a settings file, extends Matching.
    
    Public methods:
    - __init__
    """

    def __init__(self, 
                 settings_file, 
                 num_cores=None) : # pragma : no cover
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
        if num_cores is None :
            self.num_cores = multiprocessing.cpu_count()
        else :
            self.num_cores = num_cores

        try:
            self.data_model = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
            self.stop_words = pickle.load(settings_file)
        except (KeyError, AttributeError) :
            raise ValueError("This settings file is not compatible with "
                             "the current version of dedupe. This can happen "
                             "if you have recently upgraded dedupe.")
        except :
            print("Something has gone wrong with loading the settings file")
            raise
                             

        self._logLearnedWeights()
        logger.info(self.predicates)
        logger.info(self.stop_words)

        self.blocker = blocking.Blocker(self.predicates, 
                                        self.stop_words)





class ActiveMatching(Matching) :
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
                 data_sample = None,
                 num_cores = None) :
        """
        Initialize from a data model and data sample.

        #### Example usage

            # initialize from a defined set of fields
            fields = [{'field' : 'Site name', 'type': 'String'},
                      {'field' : 'Address', 'type': 'String'},
                      {'field' : 'Zip', 'type': 'String', 'Has Missing':True},
                      {'field' : 'Phone', 'type': 'String', 'Has Missing':True},
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
        self.data_model = DataModel(variable_definition)

        if num_cores is None :
            self.num_cores = multiprocessing.cpu_count()
        else :
            self.num_cores = num_cores

        self.data_sample = data_sample

        if self.data_sample :
            self._checkDataSample(self.data_sample)
            self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                         self.data_model,
                                                         self.num_cores)
        else :
            self.data_sample = []
            self.activeLearner = None

        training_dtype = [('label', 'S8'), 
                          ('distances', 'f4', 
                           (len(self.data_model['fields']), ))]

        self.training_data = numpy.zeros(0, dtype=training_dtype)
        self.training_pairs = OrderedDict({u'distinct': [], 
                                           u'match': []})

        self.learner = rlr.lr
        self.blocker = None


    def cleanupTraining(self) : # pragma : no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_data
        del self.training_pairs
        del self.activeLearner
        del self.data_sample


    def readTraining(self, training_file) :
        '''
        Read training from previously built training data file object
        
        Arguments:
        
        training_file -- file object containing the training data
        '''
        
        logger.info('reading training from file')
        
        training_pairs = json.load(training_file, 
                                   cls=serializer.dedupe_decoder)


        for (label, examples) in training_pairs.items():
            if examples :
                self._checkRecordPairType(examples[0])

            examples = core.freezeData(examples)

            training_pairs[label] = examples
            self.training_pairs[label].extend(examples)

        self._addTrainingData(training_pairs)

        self._trainClassifier(0.1)

    def train(self, ppc=.1, uncovered_dupes=1, index_predicates=True) : # pragma : no cover
        """Keyword arguments:
        ppc -- Limits the Proportion of Pairs Covered that we allow a
               predicate to cover. If a predicate puts together a fraction
               of possible pairs greater than the ppc, that predicate will
               be removed from consideration.

               As the size of the data increases, the user will generally
               want to reduce ppc.

               ppc should be a value between 0.0 and 1.0

        uncovered_dupes -- The number of true dupes pairs in our training
                           data that we can accept will not be put into any
                           block. If true true duplicates are never in the
                           same block, we will never compare them, and may
                           never declare them to be duplicates.

                           However, requiring that we cover every single
                           true dupe pair may mean that we have to use
                           blocks that put together many, many distinct pairs
                           that we'll have to expensively, compare as well.

        index_predicates -- Should dedupe consider predicates that
                            rely upon indexing the data. Index predicates can 
                            be slower and take susbstantial memory.

                            Defaults to True.

        """
        self._trainClassifier()
        self._trainBlocker(ppc, uncovered_dupes, index_predicates)

    def _trainClassifier(self, alpha=None) : # pragma : no cover

        if alpha is None :
            alpha = self._regularizer()

        self.data_model = core.trainModel(self.training_data,
                                          self.data_model, 
                                          self.learner,
                                          alpha)

        self._logLearnedWeights()

    def _regularizer(self) :
        n_folds = min(numpy.sum(self.training_data['label']==u'match')/3,
                      20)
        n_folds = max(n_folds,
                      2)

        logger.info('%d folds', n_folds)

        alpha = crossvalidation.gridSearch(self.training_data,
                                           self.learner,
                                           self.data_model, 
                                           self.num_cores,
                                           k=n_folds)

        return alpha


    
    def _trainBlocker(self, ppc=1, uncovered_dupes=1, index_predicates=True) : # pragma : no cover
        training_pairs = copy.deepcopy(self.training_pairs)

        confident_nonduplicates = training.semiSupervisedNonDuplicates(self.data_sample,
                                                                       self.data_model,
                                                                       sample_size=32000)

        training_pairs[u'distinct'].extend(confident_nonduplicates)

        predicate_set = predicateGenerator(self.data_model, 
                                           index_predicates,
                                           self.canopies)

        (self.predicates, 
         self.stop_words) = dedupe.training.blockTraining(training_pairs,
                                                          predicate_set,
                                                          ppc,
                                                          uncovered_dupes,
                                                          self._linkage_type)

        self.blocker = blocking.Blocker(self.predicates,
                                        self.stop_words) 


    def writeSettings(self, file_obj): # pragma : no cover
        """
        Write a settings file containing the 
        data model and predicates to a file object

        Keyword arguments:
        file_obj -- file object to write settings data into
        """

        pickle.dump(self.data_model, file_obj)
        pickle.dump(self.predicates, file_obj)
        pickle.dump(dict(self.stop_words), file_obj)

    def writeTraining(self, file_obj): # pragma : no cover
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


    def uncertainPairs(self) :
        '''
        Provides a list of the pairs of records that dedupe is most
        curious to learn if they are matches or distinct.
        
        Useful for user labeling.

        '''
        
        
        if self.training_data.shape[0] == 0 :
            rand_int = random.randint(0, len(self.data_sample)-1)
            random_pair = self.data_sample[rand_int]
            exact_match = (random_pair[0], random_pair[0]) 
            self._addTrainingData({u'match':[exact_match, exact_match],
                                   u'distinct':[]})


        self._trainClassifier(0.1)

        bias = len(self.training_pairs[u'match'])
        if bias :
            bias /= (bias
                     + len(self.training_pairs[u'distinct']))

        min_examples = min(len(self.training_pairs[u'match']),
                           len(self.training_pairs[u'distinct']))

        regularizer = 10 

        bias = ((0.5 * min_examples + bias * regularizer)
                /(min_examples + regularizer))

        return self.activeLearner.uncertainPairs(self.data_model, bias)

    def markPairs(self, labeled_pairs) :
        '''
        Add a labeled pairs of record to dedupes training set and update the
        matching model
        
        Argument :

        labeled_pairs -- A dictionary with two keys, `match` and `distinct`
                         the values are lists that can contain pairs of records
                         
        '''
        try :
            labeled_pairs.items()
            labeled_pairs[u'match']
            labeled_pairs[u'distinct']
        except :
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs[u'match'] :
            pair = labeled_pairs[u'match'][0]
            self._checkRecordPairType(pair)
        
        if labeled_pairs[u'distinct'] :
            pair = labeled_pairs[u'distinct'][0]
            self._checkRecordPairType(pair)
        
        if not labeled_pairs[u'distinct'] and not labeled_pairs[u'match'] :
            warnings.warn("Didn't return any labeled record pairs")
        

        for label, pairs in labeled_pairs.items() :
            self.training_pairs[label].extend(core.freezeData(pairs))

        self._addTrainingData(labeled_pairs) 



    def _checkRecordPairType(self, record_pair) :
        try :
            record_pair[0]
        except :
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs (ordered sequences of length 2)")

        if len(record_pair) != 2 :
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs")
        try :
            record_pair[0].keys() and record_pair[1].keys()
        except :
            raise ValueError("A pair of record_pairs must be made up of two "
                             "dictionaries ")

        self._checkRecordType(record_pair[0])
        self._checkRecordType(record_pair[1])

    def  _checkDataSample(self, data_sample) :
        try :
            len(data_sample)
        except TypeError :
            raise ValueError("data_sample must be a sequence")

        if len(data_sample) :
            self._checkRecordPairType(data_sample[0])

        else :
            warnings.warn("You submitted an empty data_sample")




    def _addTrainingData(self, labeled_pairs) :
        """
        Appends training data to the training data collection.
        """
    
        for label, examples in labeled_pairs.items () :
            n_examples = len(examples)
            labels = [label] * n_examples

            new_data = numpy.empty(n_examples,
                                   dtype=self.training_data.dtype)

            new_data['label'] = labels
            new_data['distances'] = core.fieldDistances(examples, 
                                                        self.data_model)

            self.training_data = numpy.append(self.training_data, 
                                              new_data)


    def _loadSample(self, data_sample) :

        self._checkDataSample(data_sample) 

        self.data_sample = data_sample

        self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                     self.data_model,
                                                     self.num_cores)



class StaticDedupe(DedupeMatching, StaticMatching) :
    """
    Mixin Class for Static Deduplication
    """

class Dedupe(DedupeMatching, ActiveMatching) :
    """
    Mixin Class for Active Learning Deduplication
    
    Public Methods
    - sample
    """
    canopies = True

    def sample(self, data, sample_size=15000, 
               blocked_proportion=0.5) :
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

        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(predicateGenerator(self.data_model, 
                                             index_predicates=False,
                                             canopies=self.canopies))


        data = sampling.randomDeque(data)
        blocked_sample_keys = sampling.dedupeBlockedSample(blocked_sample_size,
                                                           predicates,
                                                           data)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = set(dedupe.core.randomPairs(len(data),
                                                         random_sample_size))
        data = dict(data)

        data_sample = ((data[k1], data[k2])
                       for k1, k2 
                       in blocked_sample_keys | random_sample_keys)

        data_sample = core.freezeData(data_sample)

        self._loadSample(data_sample)


class StaticRecordLink(RecordLinkMatching, StaticMatching) :
    """
    Mixin Class for Static Record Linkage
    """

class RecordLink(RecordLinkMatching, ActiveMatching) :
    """
    Mixin Class for Active Learning Record Linkage
    
    Public Methods
    - sample
    """
    canopies = False

    def sample(self, data_1, data_2, sample_size=150000, 
               blocked_proportion=.5) :
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
            raise ValueError('Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError('Dictionary of records from second dataset is empty.')

        if len(data_1) > len(data_2) :
            data_1, data_2 = data_2, data_1

        data_1 = core.index(data_1)

        offset = len(data_1)
        data_2 = core.index(data_2, offset)

        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(predicateGenerator(self.data_model, 
                                             index_predicates=False,
                                             canopies=self.canopies))

        data_1 = sampling.randomDeque(data_1)
        data_2 = sampling.randomDeque(data_2)

        blocked_sample_keys = sampling.linkBlockedSample(blocked_sample_size,
                                                         predicates,
                                                         data_1, 
                                                         data_2)
        
        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = dedupe.core.randomPairsMatch(len(data_1),
                                                          len(data_2), 
                                                          random_sample_size)

        random_sample_keys = {(a, b + offset) 
                              for a, b in random_sample_keys}

        data_1 = dict(data_1)
        data_2 = dict(data_2)
        
        data_sample = ((data_1[k1], data_2[k2])
                       for k1, k2 
                       in blocked_sample_keys | random_sample_keys)

        data_sample = core.freezeData(data_sample)

        self._loadSample(data_sample)


class GazetteerMatching(RecordLinkMatching) :
    
    def __init__(self, *args, **kwargs) :
        super(GazetteerMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching
        self._linkage_type = "GazetteerMatching"
        self.blocked_records = OrderedDict({})


    def _blockData(self, messy_data) :
        for each in self._blockGenerator(messy_data, self.blocked_records) :
            yield each


    def index(self, data) : # pragma : no cover

        for field in self.blocker.index_fields :
            self.blocker.index((record[field]
                                for record 
                                in viewvalues(data)),
                               field)

        for block_key, record_id in self.blocker(data.items()) :
            if block_key not in self.blocked_records :
                self.blocked_records[block_key] = {}
            self.blocked_records[block_key][record_id] = data[record_id]

    def unindex(self, data) : # pragma : no cover

        for field in self.blocker.index_fields :
            self.blocker.unindex((record[field]
                                  for record 
                                  in viewvalues(data)),
                                 field)

        for block_key, record_id in self.blocker(viewitems(data)) :
            try : 
                del self.blocked_records[block_key][record_id]
            except KeyError :
                pass 
        
    def match(self, messy_data, threshold = 0.5, n_matches = 1) : # pragma : no cover
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

class Gazetteer(RecordLink, GazetteerMatching):
    pass

class StaticGazetteer(StaticRecordLink, GazetteerMatching):
    pass

def predicateGenerator(data_model, index_predicates, canopies) :
    predicates = set()
    for definition in data_model.primary_fields :
        for predicate in definition.predicates :
            if hasattr(predicate, 'index') :
                if index_predicates :
                    if hasattr(predicate, 'canopy') :
                        if canopies :
                            predicates.add(predicate)
                    else :
                        if not canopies :
                            predicates.add(predicate)
            else :
                predicates.add(predicate)

    return predicates

