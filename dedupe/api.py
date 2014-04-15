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
import random
import warnings
import copy
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
import dedupe.tfidf as tfidf
from dedupe.datamodel import DataModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

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
                                           self.num_processes)['score']

        probability.sort()
        probability = probability[::-1]

        expected_dupes = numpy.cumsum(probability)

        recall = expected_dupes / expected_dupes[-1]
        precision = expected_dupes / numpy.arange(1, len(expected_dupes) + 1)

        score = recall * precision / (recall + recall_weight ** 2 * precision)

        i = numpy.argmax(score)

        LOGGER.info('Maximum expected recall and precision')
        LOGGER.info('recall: %2.3f', recall[i])
        LOGGER.info('precision: %2.3f', precision[i])
        LOGGER.info('With threshold: %2.3f', probability[i])

        return probability[i]

    def matchBlocks(self, blocks, threshold=.5):
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
                                            self.num_processes,
                                            threshold)

        clusters = self._cluster(self.matches, cluster_threshold)
        
        return clusters

    def _checkRecordType(self, record) :
        for k in self.data_model.comparison_fields :
            if k not in record :
                raise ValueError("Records do not line up with data model. "
                                 "The field '%s' is in data_model but not "
                                 "in a record" % k)

    def _blockedPairs(self, blocks) :
        """
        Generate tuples of pairs of records from a block of records
        
        Arguments:
        
        blocks -- an iterable sequence of blocked records
        """
        
        block, blocks = core.peek(blocks)
        self._checkBlock(block)

        def pair_gen() :
            disjoint = set.isdisjoint
            blockPairs = self._blockPairs
            for block in blocks :
                for pair in blockPairs(block) :
                    ((key_1, record_1, smaller_ids_1), 
                     (key_2, record_2, smaller_ids_2)) = pair
                    if disjoint(smaller_ids_1, smaller_ids_2) :
                        yield (key_1, record_1), (key_2, record_2)

        return pair_gen()

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
        of record ids, where the record_ids within each tuple should refer
        to the same entity
        
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

    def _blockPairs(self, block) :  # pragma : no cover
        return itertools.combinations(block, 2)
        
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

        for field in self.blocker.tfidf_fields :
            self.blocker.tfIdfBlock(((record_id, record[field])
                                     for record_id, record 
                                     in data_d.iteritems()),
                                    field)

        for block_key, record_id in self.blocker(data_d.iteritems()) :
            blocks.setdefault(block_key, []).append((record_id, 
                                                     data_d[record_id]))

        # Redundant-free Comparisons from Kolb et al, "Dedoop:
        # Efficient Deduplication with Hadoop"
        # http://dbs.uni-leipzig.de/file/Dedoop.pdf
        for block_id, (block, records) in enumerate(blocks.iteritems()) :
            for record_id, record in records :
                coverage.setdefault(record_id, []).append(block_id)

        for block_id, (block_key, records) in enumerate(blocks.iteritems()) :
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
        Identifies pairs of records that refer to the same entity, returns tuples
        of record ids, where both record_ids within a tuple should refer
        to the same entity
        
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

    def _blockPairs(self, block) : # pragma : no cover
        base, target = block
        return itertools.product(base, target)
        
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


        for block_key, record_id in self.blocker(data_1.iteritems()) :
            blocks.setdefault(block_key, ([], []))[0].append((record_id, 
                                                              data_1[record_id]))

        for block_key, record_id in self.blocker(data_2.iteritems()) :
            if block_key in blocks :
                blocks[block_key][1].append((record_id, data_2[record_id]))

        for block_id, (_, sources) in enumerate(blocks.iteritems()) :
            for source in sources :
                for record_id, record in source :
                    coverage.setdefault(record_id, []).append(block_id)

        for block_id, (block_key, sources) in enumerate(blocks.iteritems()) :
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
                 num_processes=1) :
        """
        Initialize from a settings file
        #### Example usage

            # initialize from a settings file
            deduper = dedupe.Dedupe('my_learned_settings')

        #### Keyword arguments
        
        `settings_file`
        A file location for a settings file.


        Settings files are typically generated by saving the settings
        learned from ActiveMatching. If you need details for this
        file see the method [`writeSettings`][[api.py#writesettings]].
        """
        super(StaticMatching, self).__init__()


        if settings_file.__class__ is not str :
            raise ValueError("Must supply a settings file name")

        self.num_processes = num_processes

        with open(settings_file, 'rb') as f: # pragma : no cover
            try:
                self.data_model = pickle.load(f)
                self.predicates = pickle.load(f)
                self.stop_words = pickle.load(f)
            except KeyError :
                raise ValueError("The settings file doesn't seem to be in "
                                 "right format. You may want to delete the "
                                 "settings file and try again")


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
    """

    def __init__(self, 
                 field_definition, 
                 data_sample = None,
                 num_processes = 1) :
        """
        Initialize from a data model and data sample.

        #### Example usage

            # initialize from a defined set of fields
            fields = {'Site name': {'type': 'String'},
                      'Address':   {'type': 'String'},
                      'Zip':       {'type': 'String', 'Has Missing':True},
                      'Phone':     {'type': 'String', 'Has Missing':True},
                      }

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
        A field definition is a dictionary where the keys are the fields
        that will be used for training a model and the values are the
        field specification

        Field types include

        - String

        A 'String' type field must have as its key a name of a field
        as it appears in the data dictionary and a type declaration
        ex. `{'Phone': {type: 'String'}}`

        Longer example of a field definition:


            fields = {'name':       {'type': 'String'},
                      'address':    {'type': 'String'},
                      'city':       {'type': 'String'},
                      'cuisine':    {'type': 'String'}
                      }

        In the data_sample, each element is a tuple of two
        records. Each record is, in turn, a tuple of the record's key and
        a record dictionary.

        In in the record dictionary the keys are the names of the
        record field and values are the record values.
        """
        super(ActiveMatching, self).__init__()

        if field_definition.__class__ is not dict :
            raise ValueError('Incorrect Input Type: must supply '
                             'a field definition.')

        self.data_model = DataModel(field_definition)

        self.data_sample = data_sample

        if self.data_sample :
            self._checkDataSample(self.data_sample)
            self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                         self.data_model)
        else :
            self.activeLearner = None

        self.num_processes = num_processes


        training_dtype = [('label', 'S8'), 
                          ('distances', 'f4', 
                           (len(self.data_model['fields']), ))]

        self.training_data = numpy.zeros(0, dtype=training_dtype)
        self.training_pairs = dedupe.backport.OrderedDict({'distinct': [], 
                                                           'match': []})



    def readTraining(self, training_source) : # pragma : no cover
        '''
        Read training from previously saved training data file
        
        Arguments:
        
        training_source -- the path of the training data file
        '''

        LOGGER.info('reading training from file')

        with open(training_source, 'r') as f:
            training_pairs = json.load(f, 
                                       cls=serializer.dedupe_decoder)

        for (label, examples) in training_pairs.items():
            if examples :
                self._checkRecordPairType(examples[0])

            examples = core.freezeData(examples)

            training_pairs[label] = examples
            self.training_pairs[label].extend(examples)

        self._addTrainingData(training_pairs)

        self._trainClassifier()

    def train(self, ppc=1, uncovered_dupes=1) :
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

        LOGGER.info('%d folds', n_folds)

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

        blocker_types = self._blockerTypes()

        confident_nonduplicates = training.semiSupervisedNonDuplicates(self.data_sample,
                                                                       self.data_model,
                                                                       sample_size=32000)

        training_pairs['distinct'].extend(confident_nonduplicates)

        predicate_set = blocking.predicateGenerator(blocker_types, 
                                                    self.data_model)

        (self.predicates, 
         self.stop_words) = dedupe.blocking.blockTraining(training_pairs,
                                                          predicate_set,
                                                          ppc,
                                                          uncovered_dupes,
                                                          self._linkage_type)

        self.blocker = self._Blocker(self.predicates,
                                     self.stop_words) 


    def _blockerTypes(self) : # pragma : no cover
        string_predicates = (predicates.wholeFieldPredicate,
                             predicates.tokenFieldPredicate,
                             predicates.commonIntegerPredicate,
                             predicates.sameThreeCharStartPredicate,
                             predicates.sameFiveCharStartPredicate,
                             predicates.sameSevenCharStartPredicate,
                             predicates.nearIntegersPredicate,
                             predicates.commonFourGram,
                             predicates.commonSixGram)

        tfidf_string_predicates = tuple([tfidf.TfidfPredicate(threshold)
                                         for threshold
                                         in [0.2, 0.4, 0.6, 0.8]])

        return {'String' : (string_predicates
                            + tfidf_string_predicates)}




    def writeSettings(self, file_name): # pragma : no cover
        """
        Write a settings file that contains the 
        data model and predicates

        Keyword arguments:
        file_name -- path to file
        """

        with open(file_name, 'w') as f:
            pickle.dump(self.data_model, f)
            pickle.dump(self.predicates, f)
            pickle.dump(self.stop_words, f)

    def writeTraining(self, file_name): # pragma : no cover
        """
        Write to a json file that contains labeled examples

        Keyword arguments:
        file_name -- path to a json file
        """

        with open(file_name, 'wb') as f:
            json.dump(self.training_pairs, 
                      f, 
                      default=serializer._to_json)


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


    def _logLearnedWeights(self): # pragma: no cover
        """
        Log learned weights and bias terms
        """
        LOGGER.info('Learned Weights')
        for (key_1, value_1) in self.data_model.items():
            try:
                for (key_2, value_2) in value_1.items():
                    LOGGER.info((key_2, value_2['weight']))
            except AttributeError:
                LOGGER.info((key_1, value_1))

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




