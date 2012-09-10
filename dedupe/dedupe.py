#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random

import core
import training_sample
import crossvalidation
from predicates import *
import blocking
import clustering
import numpy


def sampleDict(d, sample_size):

    if len(d) <= sample_size:
        return d

    sample_keys = random.sample(d.keys(), sample_size)
    return dict((k, d[k]) for k in d.keys() if k in sample_keys)


class Dedupe:

    def __init__(self, init=None, input_type=None):
        if init:
            if input_type == 'fields':
                self.initializeSettings(init)
            elif input_type == 'settings file':
                self.readSettings(init)
            elif input_type:
                raise ValueError("Invalid Input Type, input_type "
                                 "must be 'fields' or 'settings file'"
                                 )
            else:
                raise ValueError("Undefined Input Type, input_type "
                                 "must be 'fields' or 'settings file'"
                                 )

    def initializeSettings(self, fields):
        self.predicates = None
        self.alpha = 0
        self.data_model = {}
        self.data_model['fields'] = {}

        for (k, v) in fields.iteritems():
            v.update({'weight': 0})
            self.data_model['fields'][k] = v

        self.data_model['bias'] = 0

        field_dtype = [('names', 'a20', len(fields)), ('values', 'f4',
                       len(fields))]

        training_dtype = [('label', 'i4'), ('field_distances',
                          field_dtype)]
        self.training_data = numpy.zeros(0, dtype=training_dtype)


    def trainingDistance(self):
        self.training_data = \
            training_sample.addTrainingData(self.training_pairs,
                                            self.data_model,
                                            self.training_data)

    def findAlpha(self):
        self.alpha = crossvalidation.gridSearch(self.training_data,
                                                core.trainModel,
                                                self.data_model, k=10)

    def train(self):
        self.findAlpha()
        self.data_model = core.trainModel(self.training_data,
                                          self.data_model,
                                          self.alpha)
        self.printLearnedWeights()

    def activeLearning(self,
                       data_d,
                       labelingFunction,
                       numTrainingPairs=30,
                       ):

        (self.training_data,
         self.training_pairs,
         self.data_model) = training_sample.activeLearning(sampleDict(data_d,
                                                                      700),
                                                           self.data_model,
                                                           labelingFunction,
                                                           numTrainingPairs,
                                                           self.training_data)
        self.train()

    def learnBlocking(self, data_d, semi_supervised=True):
        if semi_supervised:
            confident_nonduplicates = blocking.semiSupervisedNonDuplicates(sampleDict(data_d, 700),
                                                                           self.data_model)

            self.training_pairs[0].extend(confident_nonduplicates)

        predicate_functions = (wholeFieldPredicate,
                               tokenFieldPredicate,
                               commonIntegerPredicate,
                               sameThreeCharStartPredicate,
                               sameFiveCharStartPredicate,
                               sameSevenCharStartPredicate,
                               nearIntegersPredicate,
                               commonFourGram,
                               commonSixGram,
                               )

        blocker = blocking.Blocking(self.training_pairs,
                                    predicate_functions,
                                    self.data_model)

        self.predicates = blocker.trainBlocking()

    def mapBlocking(self, data_d, semi_supervised=True):
        self.blocked_map = blocking.blockingIndex(data_d,
                                                  self.predicates)

    def identifyCandidates(self):
        self.candidates = blocking.mergeBlocks(self.blocked_map)

    def score(self, data_d, threshold=None):
        self.dupes = core.scoreDuplicates(self.candidates,
                                          data_d,
                                          self.data_model,
                                          threshold)

    def findDuplicates(self,
                       data_d,
                       semi_supervised=True,
                       threshold=None,
                       ):

        if not self.predicates:
            self.learnBlocking(data_d, semi_supervised)

        self.mapBlocking(data_d)
        self.identifyCandidates()
        self.printBlockingSummary(data_d)
        print 'finding duplicates ...'
        self.score(data_d, threshold)

    def duplicateClusters(self,
                          clustering_algorithm=clustering.hierarchical.cluster,
                          **args):
        return clustering_algorithm(self.dupes, **args)

    def printLearnedWeights(self):
        print 'Learned Weights'
        for (k1, v1) in self.data_model.items():
            try:
                for (k2, v2) in v1.items():
                    print (k2, v2['weight'])
            except:
                print (k1, v1)

    def printBlockingSummary(self, data_d):
        print 'Blocking reduced the number of comparisons by',
        print int((1 - len(self.candidates) / float(0.5 * len(data_d) ** 2)) * 100),
        print '%'
        print "We'll make",
        print len(self.candidates),
        print 'comparisons.'

    def writeSettings(self, file_name):
        source_predicates = []
        for predicate_tuple in self.predicates:
            source_predicate = []
            for predicate in predicate_tuple:
                source_predicate.append((predicate[0].__name__,
                                         predicate[1]))
            source_predicates.append(source_predicate)

        with open(file_name, 'w') as f:
            json.dump({'data model': self.data_model,
                      'predicates': source_predicates}, f)

    def readSettings(self, file_name):
        with open(file_name, 'r') as f:
            learned_settings = json.load(f)

        self.data_model = learned_settings['data model']
        self.predicates = []
        for predicate_l in learned_settings['predicates']:
            predicate_tuple = tuple([(eval(predicate[0]), predicate[1])
                                    for predicate in predicate_l])
            self.predicates.append(predicate_tuple)

    def writeTraining(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.training_pairs, f)

    def readTraining(self, file_name):
        with open(file_name, 'r') as f:
            training_pairs_raw = json.load(f)

        training_pairs = {0: [], 1: []}
        for (label, examples) in training_pairs_raw.iteritems():
            for pair in examples:
                training_pairs[int(label)].append((core.frozendict(pair[0]),
                                                   core.frozendict(pair[1])))

        self.training_pairs = training_pairs
        self.trainingDistance()
