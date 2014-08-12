#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""

try:
    import json
except ImportError: 
    import simplejson as json
import itertools
import logging
import pickle
import numpy
import multiprocessing
import random
import warnings
import copy
import os

try:
    from collections import OrderedDict
except ImportError :
    from dedupe.backport import OrderedDict

import dedupe
import dedupe.core as core
import dedupe.training as training
import dedupe.serializer as serializer
import dedupe.crossvalidation as crossvalidation
import dedupe.predicates as predicates
import dedupe.blocking as blocking
import dedupe.clustering as clustering
from dedupe.datamodel import DataModel
import dedupe.centroid as centroid

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
        self.matches = None
        self.blocker = None

    def __del__(self) :
        try :
            os.remove(self.matches.filename)
        except :
            pass

    def thresholdBlocks(self, blocks, recall_weight=1.5):
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

    def matchBlocks(self, blocks, threshold=.5, *args, **kwargs):
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
        
        self.matches = core.scoreDuplicates(candidate_records,
                                            self.data_model,
                                            self.num_cores,
                                            threshold)

        logger.info("matching done, begin clustering")

        clusters = self._cluster(self.matches, 
                                 cluster_threshold, *args, **kwargs)
        
        return clusters

    def _checkRecordType(self, record) :
        for field, _ in self.data_model.field_comparators :
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
            except TypeError:
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
        self._Blocker = blocking.DedupeBlocker
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
        if not block :
            raise ValueError("You have not provided any data blocks")
        try :
            if len(block[0]) < 3 :
                raise ValueError("Each item in a block must be a "
                                 "sequence of record_id, record, and smaller ids "
                                 "and the records also must be dictionaries")
        except :
            raise ValueError("sandwich Each item in a block must be a "
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

        blocks = OrderedDict({})
        coverage = {}

        key_index = {}
        indexed_data = {}
        for i, (key, value) in enumerate(data_d.iteritems()) :
            key_index[i] = key
            indexed_data[i] = value
            
        for field in self.blocker.tfidf_fields :
            self.blocker.tfIdfBlock(((record_id, record[field])
                                     for record_id, record 
                                     in indexed_data.iteritems()),
                                    field)

        for block_key, record_id in self.blocker(indexed_data.iteritems()) :
            blocks.setdefault(block_key, []).append(record_id) 

        blocks = blocks.values()

        blocks = [records for records in blocks if len(records) > 1]

        blocks = [[(key_index[record_id], indexed_data[record_id])
                   for record_id in records]
                  for records in blocks]

        # Redundant-free Comparisons from Kolb et al, "Dedoop:
        # Efficient Deduplication with Hadoop"
        # http://dbs.uni-leipzig.de/file/Dedoop.pdf
        for block_id, records in enumerate(blocks) :
            for record_id, record in records :
                coverage.setdefault(record_id, []).append(block_id)

        blocks = iter(blocks)

        for block_id, records in enumerate(blocks) :
            if block_id % 10000 == 0 :
                logger.info("%s blocks" % block_id)
            tuple_records = []
            for record_id, record in records :
                smaller_ids = set([covered_id for covered_id 
                                   in coverage[record_id] 
                                   if covered_id < block_id])
                tuple_records.append((record_id, record, smaller_ids))

            yield tuple_records


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
        self._Blocker = blocking.RecordLinkBlocker
        self._linkage_type = "RecordLink"

    def match(self, data_1, data_2, threshold = 1.5) : # pragma : no cover
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
        data_1        --  Dictionary of records from first dataset, where the keys 
                          are record_ids and the values are dictionaries with the keys 
                          being field names

        data_2        --  Dictionary of records from second dataset, same form as data_1

        recall_weight -- Sets the tradeoff between precision and
                         recall. I.e. if you care twice as much about
                         recall as you do precision, set recall_weight
                         to 2.
        """

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
        try :
            base, target = block
        except :
            raise ValueError("Each block must be a made up of two "
                             "sequences, (base_sequence, target_sequence)")

        if base :
            if len(base[0]) < 3 :
                raise ValueError("Each block must be a made up of two "
                                 "sequences, (base_sequence, target_sequence)")
            self._checkRecordType(base[0][1])
        if target :
            if len(target[0]) < 3 :
                raise ValueError("Each block must be a made up of two "
                                 "sequences, (base_sequence, target_sequence)")
                
            self._checkRecordType(target[0][1])

    def _blockData(self, data_1, data_2) :

        blocks = OrderedDict({})
        coverage = {}

        for field in self.blocker.tfidf_fields :
            fields_1 = ((record_id, record[field])
                        for record_id, record 
                        in data_1.iteritems())
            fields_2 = ((record_id, record[field])
                        for record_id, record 
                        in data_2.iteritems())

            self.blocker.tfIdfBlock(fields_1, fields_2, field)


        for block_key, record_id in self.blocker(data_2.items()) :
            blocks.setdefault(block_key, ([], []))[1].append((record_id, 
                                                              data_2[record_id]))
        for block_key, record_id in self.blocker(data_1.items()) :
            if block_key in blocks :
                blocks[block_key][0].append((record_id, data_1[record_id]))

        blocks = blocks.values()

        for block_id, sources in enumerate(blocks) :
            for source in sources :
                for record_id, record in source :
                    coverage.setdefault(record_id, []).append(block_id)

        for block_id, sources in enumerate(blocks) :
            if block_id % 10000 == 0 :
                logger.info("%s blocks" % block_id)
            tuple_block = []
            for source in sources :
                tuple_source = []
                for record_id, record in source :
                    smaller_ids = set([covered_id for covered_id 
                                       in coverage[record_id] 
                                       if covered_id < block_id])
                    tuple_source.append((record_id, record, smaller_ids))
                tuple_block.append(tuple_source)

            yield tuple_block

class StaticMatching(Matching) :
    """
    Class for initializing a dedupe object from a settings file, extends Matching.
    
    Public methods:
    - __init__
    """

    def __init__(self, 
                 settings_file, 
                 num_cores=None) :
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
        super(StaticMatching, self).__init__()

        if num_cores is None :
            self.num_cores = multiprocessing.cpu_count()
        else :
            self.num_cores = num_cores

        try:
            self.data_model = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
            self.stop_words = pickle.load(settings_file)
        except KeyError :
            raise ValueError("The settings file doesn't seem to be in "
                             "right format. You may want to delete the "
                             "settings file and try again")

        self._logLearnedWeights()
        logger.info(self.predicates)
        logger.info(self.stop_words)



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
        super(ActiveMatching, self).__init__()

        self.data_model = DataModel(variable_definition)

        self.data_sample = data_sample

        if self.data_sample :
            self._checkDataSample(self.data_sample)
            self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                         self.data_model)
        else :
            self.activeLearner = None

        if num_cores is None :
            self.num_cores = multiprocessing.cpu_count()
        else :
            self.num_cores = num_cores

        training_dtype = [('label', 'S8'), 
                          ('distances', 'f4', 
                           (len(self.data_model['fields']), ))]

        self.training_data = numpy.zeros(0, dtype=training_dtype)
        self.training_pairs = dedupe.backport.OrderedDict({'distinct': [], 
                                                           'match': []})


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

        self._trainClassifier()

    def train(self, ppc=.1, uncovered_dupes=1) :
        """
        Keyword arguments:
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
        """
        n_folds = min(numpy.sum(self.training_data['label']=='match')/3,
                      20)
        n_folds = max(n_folds,
                      2)

        logger.info('%d folds', n_folds)

        alpha = crossvalidation.gridSearch(self.training_data,
                                           core.trainModel, 
                                           self.data_model, 
                                           k=n_folds)


        self._trainClassifier(alpha)
        self._trainBlocker(ppc, uncovered_dupes)


    def _trainClassifier(self, alpha=.1) : # pragma : no cover

        self.data_model = core.trainModel(self.training_data,
                                          self.data_model, 
                                          alpha)

        self._logLearnedWeights()

    
    def _trainBlocker(self, ppc=1, uncovered_dupes=1) :
        training_pairs = copy.deepcopy(self.training_pairs)

        confident_nonduplicates = training.semiSupervisedNonDuplicates(self.data_sample,
                                                                       self.data_model,
                                                                       sample_size=32000)

        training_pairs['distinct'].extend(confident_nonduplicates)

        predicate_set = predicateGenerator(self.data_model)

        (self.predicates, 
         self.stop_words) = dedupe.blocking.blockTraining(training_pairs,
                                                          predicate_set,
                                                          ppc,
                                                          uncovered_dupes,
                                                          self._linkage_type)

        self.blocker = self._Blocker(self.predicates,
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
                  ensure_ascii=False)


    def uncertainPairs(self) :
        '''
        Provides a list of the pairs of records that dedupe is most curious to learn 
        if they are matches or distinct.
        
        Useful for user labeling.
        '''
        
        
        if self.training_data.shape[0] == 0 :
            rand_int = random.randint(0, len(self.data_sample))
            random_pair = self.data_sample[rand_int]
            exact_match = (random_pair[0], random_pair[0]) 
            self._addTrainingData({'match':[exact_match, exact_match],
                                   'distinct':[]})


            self._trainClassifier(alpha=0.1)

        
        dupe_ratio = (len(self.training_pairs['match'])
                      /(len(self.training_pairs['distinct']) + 1.0))

        return self.activeLearner.uncertainPairs(self.data_model, dupe_ratio)

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
            labeled_pairs['match']
            labeled_pairs['distinct']
        except :
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs['match'] :
            pair = labeled_pairs['match'][0]
            self._checkRecordPairType(pair)
        
        if labeled_pairs['distinct'] :
            pair = labeled_pairs['distinct'][0]
            self._checkRecordPairType(pair)
        
        if not labeled_pairs['distinct'] and not labeled_pairs['match'] :
            warnings.warn("Didn't return any labeled record pairs")
        

        for label, pairs in labeled_pairs.items() :
            self.training_pairs[label].extend(core.freezeData(pairs))

        self._addTrainingData(labeled_pairs) 

        self._trainClassifier(alpha=.1)



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



    def _loadSample(self, *args, **kwargs) : # pragma : no cover

        data_sample = self._sample(*args, **kwargs)

        self._checkDataSample(data_sample) 

        self.data_sample = data_sample

        self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                     self.data_model)



class StaticDedupe(DedupeMatching, StaticMatching) :
    """
    Mixin Class for Static Deduplication
    """

    def __init__(self, *args, **kwargs) :
        super(StaticDedupe, self).__init__(*args, **kwargs)

        self.blocker = self._Blocker(self.predicates, 
                                     self.stop_words)

class Dedupe(DedupeMatching, ActiveMatching) :
    """
    Mixin Class for Active Learning Deduplication
    
    Public Methods
    - sample
    """

    
    def sample(self, data, sample_size=150000) :
        '''
        Draw a random sample of combinations of records from 
        the the dataset, and initialize active learning with this sample
        
        Arguments:
        data        -- Dictionary of records, where the keys are record_ids 
                       and the values are dictionaries with the keys being 
                       field names
        
        sample_size -- Size of the sample to draw
        '''
        
        self._loadSample(data, sample_size)

    def _sample(self, data, sample_size) :

        indexed_data = dict((i, dedupe.core.frozendict(v)) 
                            for i, v in enumerate(data.values()))

        random_pairs = dedupe.core.randomPairs(len(indexed_data), 
                                               sample_size)

        data_sample = tuple((indexed_data[int(k1)], 
                             indexed_data[int(k2)]) 
                            for k1, k2 in random_pairs)

        return data_sample




class StaticRecordLink(RecordLinkMatching, StaticMatching) :
    """
    Mixin Class for Static Record Linkage
    """

    def __init__(self, *args, **kwargs) :
        super(StaticRecordLink, self).__init__(*args, **kwargs)

        self.blocker = self._Blocker(self.predicates, 
                                     self.stop_words)

class RecordLink(RecordLinkMatching, ActiveMatching) :
    """
    Mixin Class for Active Learning Record Linkage
    
    Public Methods
    - sample
    """

    def sample(self, data_1, data_2, sample_size=150000) :
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
        
        self._loadSample(data_1, data_2, sample_size)

    def _sample(self, data_1, data_2, sample_size) :

        d_1 = dict((i, dedupe.core.frozendict(v)) 
                    for i, v in enumerate(data_1.values()))
        d_2 = dict((i, dedupe.core.frozendict(v)) 
                   for i, v in enumerate(data_2.values()))

        random_pairs = dedupe.core.randomPairsMatch(len(d_1),
                                                    len(d_2), 
                                                    sample_size)
        
        data_sample = tuple((d_1[int(k1)], 
                             d_2[int(k2)]) 
                            for k1, k2 in random_pairs)

        return data_sample

class GazetteerMatching(RecordLinkMatching) :
    
    def __init__(self, *args, **kwargs) :
        super(GazetteerMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.gazetteMatching
        self._linkage_type = "GazetteerMatching"
        self.index = {}

    def match(self, data_1, data_2, threshold = 1.5, n_matches = 1) : # pragma : no cover
        """Identifies pairs of records that refer to the same entity, returns
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
        
        n_matches -- Maximum number of possible matches from data_2
                     for each record in data_1
        """
        self._cluster = clustering.gazetteMatching
        blocked_pairs = self._blockData(data_1, data_2)
        return self.matchBlocks(blocked_pairs, threshold, n_matches)

class Gazetteer(RecordLink, GazetteerMatching):
    pass

class StaticGazetteer(StaticRecordLink, GazetteerMatching):
    pass

def predicateGenerator(data_model) :
    predicates = set([])
    for definition in data_model['fields'] :
        if hasattr(definition, 'predicates') :
            predicates.update(definition.predicates)

    return predicates
