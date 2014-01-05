#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""

try:
    from json.scanner import py_make_scanner
    import json
except ImportError:
    from simplejson.scanner import py_make_scanner
    import simplejson as json
import itertools
import logging
import types
import pickle
import multiprocessing
import numpy
import random

import dedupe
import dedupe.core as core
import dedupe.training as training
import dedupe.training_serializer as training_serializer
import dedupe.crossvalidation as crossvalidation
import dedupe.predicates as predicates
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.tfidf as tfidf
from dedupe.datamodel import DataModel


class Matching(object):
    """
    Public methods:

    - `__init__`
    - `train`
    - `blockingFunction`
    - `goodThreshold`
    - `match`
    - `writeTraining`
    - `writeSettings`
    """

    def __init__(self) :
        self.matches = None
        self.blocker = None

    def goodThreshold(self, blocks, recall_weight=1.5):
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
        probability = core.scoreDuplicates(self.blockedPairs(blocks), 
                                           self.data_model, 
                                           self.pool)['score']

        probability.sort()
        probability = probability[::-1]

        expected_dupes = numpy.cumsum(probability)

        recall = expected_dupes / expected_dupes[-1]
        precision = expected_dupes / numpy.arange(1, len(expected_dupes) + 1)

        score = recall * precision / (recall + recall_weight ** 2 * precision)

        i = numpy.argmax(score)

        logging.info('Maximum expected recall and precision')
        logging.info('recall: %2.3f', recall[i])
        logging.info('precision: %2.3f', precision[i])
        logging.info('With threshold: %2.3f', probability[i])

        return probability[i]

    def match(self, blocks, threshold=.5):
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

        candidate_records = self.blockedPairs(blocks)
        
        self.matches = core.scoreDuplicates(candidate_records,
                                            self.data_model,
                                            self.pool,
                                            threshold)

        clusters = self._cluster(self.matches, cluster_threshold)
        
        return clusters

class DedupeMatching(Matching) :
    def __init__(self, *args, **kwargs) :
        super(DedupeMatching, self).__init__(*args, **kwargs)
        self._Blocker = blocking.DedupeBlocker
        self._cluster = clustering.cluster
        self._linkage_type = "Dedupe"
        

    def blockedPairs(self, blocks) :
        for block in blocks :

            block_pairs = itertools.combinations(block.items(), 2)

            for pair in block_pairs :
                yield pair


class RecordLinkMatching(Matching) :
    def __init__(self, *args, **kwargs) :
        super(RecordLinkMatching, self).__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching
        self._Blocker = blocking.RecordLinkBlocker
        self._linkage_type = "RecordLink"


    def blockedPairs(self, blocks) :
        for block in blocks :
            base, target = block
            block_pairs = itertools.product(base.items(), target.items())

            for pair in block_pairs :
                yield pair


        

class StaticMatching(Matching) :
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
        if settings_file.__class__ is not str :
            raise ValueError("Must supply a settings file name")

        self.pool = multiprocessing.Pool(processes=num_processes)

        self._readSettings(settings_file)

    def _readSettings(self, file_name):
        with open(file_name, 'rb') as f:
            try:
                self.data_model = pickle.load(f)
                self.predicates = pickle.load(f)
            except KeyError :
                raise ValueError("The settings file doesn't seem to be in "
                                 "right format. You may want to delete the "
                                 "settings file and try again")


class ActiveMatching(Matching) :
    def __init__(self, 
                 field_definition, 
                 data_sample,
                 num_processes=1) :
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
        if field_definition.__class__ is not dict :
            raise ValueError('Incorrect Input Type: must supply '
                             'a field definition.')


        self.pool = multiprocessing.Pool(processes=num_processes)

        self.data_model = DataModel(field_definition)
        self.data_sample = data_sample

        training_dtype = [('label', 'i4'), 
                          ('distances', 'f4', 
                           (len(self.data_model['fields']), ))]

        self.training_data = numpy.zeros(0, dtype=training_dtype)
        self.training_pairs = {0: [], 1: []}

        self.activeLearner = training.ActiveLearning(self.data_sample, 
                                                     self.data_model)


    def trainFromFile(self, training_source) :
        assert training_source.__class__ is str

        logging.info('reading training from file')

        self._readTraining(training_source)

        self.trainClassifier()

    # === Dedupe.trainClassifier ===
    def trainClassifier(self, alpha=None) :
        if alpha is None :

            n_folds = min(numpy.sum(self.training_data['label'])/3, 20)

            n_folds = min(max(2,
                              numpy.sum(self.training_data['label'])/3),
                          20)

            logging.info('%d folds', n_folds)

            alpha = crossvalidation.gridSearch(self.training_data,
                                               core.trainModel, 
                                               self.data_model, 
                                               k=n_folds)

        self.data_model = core.trainModel(self.training_data,
                                          self.data_model, 
                                          alpha)

        self._logLearnedWeights()

    
    # === Dedupe.trainBlocker ===
    def trainBlocker(self, ppc=1, uncovered_dupes=1) :
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

        blocker_types = self.blockerTypes()

        confident_nonduplicates = training.semiSupervisedNonDuplicates(self.data_sample,
                                                                       self.data_model,
                                                                       sample_size=32000)

        self.training_pairs[0].extend(confident_nonduplicates)


        predicate_set = blocking.predicateGenerator(blocker_types, 
                                                    self.data_model)

        
        self.predicates = dedupe.blocking.blockTraining(self.training_pairs,
                                                        predicate_set,
                                                        ppc,
                                                        uncovered_dupes,
                                                        self.pool,
                                                        self._linkage_type)

        self.blocker = self._Blocker(self.predicates, self.pool)

    def blockerTypes(self) :
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




    # === writeSettings === 

    def writeSettings(self, file_name):
        """
        Write a settings file that contains the 
        data model and predicates

        Keyword arguments:
        file_name -- path to file
        """

        with open(file_name, 'w') as f:
            pickle.dump(self.data_model, f)
            pickle.dump(self.predicates, f)

    def writeTraining(self, file_name):
        """
        Write to a json file that contains labeled examples

        Keyword arguments:
        file_name -- path to a json file
        """

        d_training_pairs = {}
        for (label, pairs) in self.training_pairs.iteritems():
            d_training_pairs[label] = [(dict(pair[0]), 
                                        dict(pair[1])) 
                                       for pair in pairs]

        with open(file_name, 'wb') as f:
            json.dump(d_training_pairs, f, default=training_serializer._to_json)


    def getUncertainPair(self) :
        if self.training_data.shape[0] == 0 :
            rand_int = random.randint(0, len(self.data_sample))
            exact_match = self.data_sample[rand_int]
            self._addTrainingData({1:[exact_match, exact_match],
                                   0:[]})


            self.trainClassifier(alpha=0.1)

        
        dupe_ratio = len(self.training_pairs[1])/(len(self.training_pairs[0]) + 1.0)

        return self.activeLearner.getUncertainPair(self.data_model, dupe_ratio)

    def markPairs(self, labeled_pairs) :
        for label, pair in labeled_pairs.items() :
            self.training_pairs[label].extend(pair)

        self._addTrainingData(labeled_pairs) 

        self.trainClassifier(alpha=.1)


    def _readTraining(self, file_name):
        """Read training pairs from a file"""
        with open(file_name, 'r') as f:
            training_pairs_raw = json.load(f, cls=training_serializer.dedupe_decoder)

        for (label, examples) in training_pairs_raw.iteritems():
            for pair in examples:
                pair = (core.frozendict(pair[0]),
                        core.frozendict(pair[1]))

                self.training_pairs[int(label)].append(pair)

        self._addTrainingData(self.training_pairs)


    def _addTrainingData(self, labeled_pairs) :
        """
        Appends training data to the training data collection.
        """
    
        nondupes, dupes = labeled_pairs[0], labeled_pairs[1]
        labels = ([0] * len(nondupes) 
                  + [1] * len(dupes))

        new_training_data = numpy.empty(len(labels),
                                        dtype=self.training_data.dtype)

        new_training_data['label'] = labels
        new_training_data['distances'] = core.fieldDistances(nondupes + dupes, 
                                                             self.data_model)

        self.training_data = numpy.append(self.training_data, 
                                          new_training_data)


    def _logLearnedWeights(self):
        """
        Log learned weights and bias terms
        """
        logging.info('Learned Weights')
        for (k1, v1) in self.data_model.items():
            try:
                for (k2, v2) in v1.items():
                    logging.info((k2, v2['weight']))
            except AttributeError:
                logging.info((k1, v1))


class StaticDedupe(DedupeMatching, StaticMatching) :
    def __init__(self, *args, **kwargs) :
        super(StaticDedupe, self).__init__(*args, **kwargs)

        self.blocker = self._Blocker(self.predicates, self.pool)

class Dedupe(DedupeMatching, ActiveMatching) :
    pass

class StaticRecordLink(RecordLinkMatching, StaticMatching) :
    def __init__(self, *args, **kwargs) :
        super(StaticRecordLink, self).__init__(*args, **kwargs)

        self.blocker = self._Blocker(self.predicates, self.pool)


class RecordLink(RecordLinkMatching, ActiveMatching) :
    pass

