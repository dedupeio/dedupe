#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import product, chain, combinations
from math import sqrt, log
import core
import tfidf
from random import sample, random, choice, shuffle
import types

class Blocker: 
    def __init__(self, predicates, df_index):
        self.predicates = predicates
        self.df_index = df_index
        self.simple_predicates = []
        self.tfidf_thresholds = []
        self.mixed_predicates = []
        self.tfidf_fields = set([])
        self.inverted_index = defaultdict(lambda: defaultdict(set))
        self.corpus_ids = set([])

        for predicate in predicates:

            if all(isinstance(pred[0], types.FunctionType)
                    for pred in predicate) :
                self.simple_predicates.append(predicate)
            elif all(pred[0].__class__ is tfidf.TfidfPredicate 
                     for pred in predicate) :
                self.tfidf_thresholds.append(predicate)
            elif all(isinstance(pred[0], types.FunctionType)
                     or pred[0].__class__ is tfidf.TfidfPredicate
                     for pred in predicate) :
                self.mixed_predicates.append(predicate)
            else:
                print predicate[0].__class__
                raise ValueError("Undefined predicate type")


        for predicate in self.tfidf_thresholds :
            for _, field in predicate :
                self.tfidf_fields.add(field)

        for predicate in self.mixed_predicates :
            for pred, field in predicate :
                if pred.__class__ is tfidf.TfidfPredicate :
                    self.tfidf_fields.add(field)


    def __call__(self, instance) :
        record_id, record = instance 
        keys = []
        for predicate in self.simple_predicates:
            key_tuples = product(*[F(record[field])
                                         for (F, field) in predicate]
                                       )
            keys.extend([str(key_tuple) for key_tuple in key_tuples])

            if (self.tfidf_thresholds):
                for field in self.tfidf_fields :
                    tokens = tfidf.getTokens(record[field])
                    for token in set(tokens) :
                      self.inverted_index[field][token].add(record_id)
                      self.corpus_ids.add(record_id)

        return keys

    def createCanopies(self, select_function, field, threshold) :
      """
      A function that returns 
      a field value of a record with a particular doc_id, doc_id
      is the only argument that must be accepted by select_function
      """

      blocked_data = []
      seen_set = set([])
      corpus_ids = self.corpus_ids.copy()

 
      while corpus_ids :
        doc_id = corpus_ids.pop()
        center = select_function(doc_id)
        # print doc_id, center
        if not center :
          continue
        
        seen_set.add(doc_id)

        # initialize the potential block with center
        block = [doc_id]
        candidate_set = set([])
        tokens = tfidf.getTokens(center)
        center_dict = tfidf.tfidfDict(center, self.df_index)

        for token in tokens :
          candidate_set.update(self.inverted_index[field][token])

        # print candidate_set
        candidate_set = candidate_set - seen_set
        for doc_id in candidate_set :
          candidate_dict = tfidf.tfidfDict(select_function(doc_id), self.df_index)
          similarity = tfidf.cosineSimilarity(candidate_dict, center_dict)

          if similarity > threshold :
            block.append(doc_id)
            seen_set.add(doc_id)
            corpus_ids.remove(doc_id)

        if len(block) > 1 :
          blocked_data.append(block)

      return blocked_data

    def canopies(self, corpus, threshold):
        return tfidf.createCanopies(corpus, self.inverted_index, threshold)

def createBlockingFunction(predicates) :

    def blockingFunction(instance) :
        keys = []
        for predicate in predicates:
            key_tuples = product(*[F(instance[field])
                                         for (F, field) in predicate]
                                       )
            keys.extend([str(key_tuple) for key_tuple in key_tuples])

        return keys

    return blockingFunction
        


def blockingIndex(data_d, blocker):

    blocked_data = defaultdict(set)
    for (key, instance) in data_d.iteritems():
        predicate_keys = blocker((key, instance)) 
        for predicate_key in predicate_keys :
            blocked_data[predicate_key].add((key, instance))

    for field in blocker.tfidf_fields :
        selector = lambda record_id : data_d[record_id][field]    
        # print field
        blocks = blocker.createCanopies(selector, field, .5)
        # print blocks
        for block in blocks:
          key = 'ID:' + str(block[0])
          for record_id in block:
            blocked_data[key].add((record_id, data_d[record_id]))            


    return blocked_data


def mergeBlocks(blocked_data):
    candidates = set()
    for block in blocked_data.values():
        block = sorted(block)
        block = [(record_id, core.frozendict(record)) for
                 record_id, record in block]
        for pair in combinations(block, 2):
            candidates.add(pair)

    return candidates


#TODO: move this to core.py
def allCandidates(data_d, key_groups=[]):
    candidates = []
    if key_groups:
        for group in key_groups :
            data_group = ((k, data_d[k]) for k in group if k in data_d)
            candidates.extend(combinations(data_group, 2))
    else:
        candidates = list(combinations(data_d.iteritems(), 2))

    return candidates
    #return list(combinations(sorted(data_d.keys()), 2))

def semiSupervisedNonDuplicates(data_d, data_model, 
                                nonduplicate_confidence_threshold=.7,
                                sample_size = 2000):


    pair_combinations = list(combinations(data_d.iteritems(), 2))

    if len(pair_combinations) <= sample_size :
        return pair_combinations

    shuffle(pair_combinations)
    
    confident_distinct_pairs = []
    n_distinct_pairs = 0
    for pair in pair_combinations :

        pair_distance = core.recordDistances([pair], data_model)
        score = core.scorePairs(pair_distance, data_model)


        if score < (1 - nonduplicate_confidence_threshold):
            key_pair, value_pair = zip(*pair)
            confident_distinct_pairs.append(value_pair)
            n_distinct_pairs += 1
            if n_distinct_pairs == sample_size :
                return confident_distinct_pairs


class Blocking:

    def __init__(self,
                 training_pairs,
                 predicate_functions,
                 data_model,
                 tfidf_thresholds = [],
                 df_index = {},
                 eta=0.01,
                 epsilon=2,
                 ):
        self.epsilon = epsilon
        self.predicate_functions = predicate_functions
        self.df_index = df_index
        self.tfidf_thresholds = [tfidf.TfidfPredicate(threshold) for threshold in tfidf_thresholds]
        self._overlap = defaultdict(int)

        self.fields = [field for field in data_model['fields']
                       if data_model['fields'][field]['type']
                       != 'Interaction']


        self.training_dupes = (training_pairs[1])[:]
        self.training_distinct = (training_pairs[0])[:]

        # We want to throw away the predicates that puts together too many
        # distinct pairs
        self.coverage_threshold = eta * len(self.training_distinct)

    # Approximate learning of blocking following the ApproxRBSetCover from
    # page 102 of Bilenko
    def trainBlocking(self, disjunctive=False):
        self.predicate_set = self.createPredicateSet(disjunctive)
        n_predicates = len(self.predicate_set)

        # TF-IDF coverage
        if self.tfidf_thresholds :
            tfidf_block_coverage = {}
            for threshold in self.tfidf_thresholds :
                for field in self.fields :
                    coverage =  tfidf.coverage(threshold.threshold, 
                                field, 
                                self.training_dupes + self.training_distinct,
                                self.df_index)
                    
                    for pair, value in coverage.iteritems():
                        self._overlap[(pair, (threshold, field))] = value


                    self.predicate_set.append(((threshold, field),))
                    
    
        #print self.predicate_set
        #print self._overlap
        found_dupes = self.predicateCoverage(self.training_dupes)
        found_distinct = self.predicateCoverage(self.training_distinct)

        ## for k,v in found_dupes.iteritems() :
        ##     print k, len(v)

        ## for k,v in found_distinct.iteritems() :
        ##     print k, len(v)

        # Only consider predicates that cover at least one duplicate pair
        self.predicate_set = found_dupes.keys()

        # We want to throw away the predicates that puts together too
        # many distinct pairs
        [self.predicate_set.remove(predicate)
         for predicate
         in found_distinct
         if len(found_distinct[predicate]) >= self.coverage_threshold]

        # Expected number of predicates that should cover a duplicate
        # pair
        expected_dupe_cover = sqrt(n_predicates
                                   / log(len(self.training_dupes)))

        found_distinct = self.filterOutIndistinctPairs(expected_dupe_cover,
                                                       found_distinct,
                                                       self.training_distinct)

        final_predicate_set = self.findOptimumBlocking(self.training_dupes,
                                                       self.predicate_set,
                                                       found_dupes,
                                                       found_distinct)

        print 'Final predicate set'
        print final_predicate_set

        if final_predicate_set:
            return final_predicate_set
        else:
            print 'No predicate found!'
            raise



    #@profile
    def predicateCoverage(self, pairs):
        coverage = defaultdict(list)
        for pair in pairs:
            for predicate in self.predicate_set:
                for basic_predicate in predicate :
                    if self._overlap[(pair, basic_predicate)] == -1 :
                        break
                    if self._overlap[(pair, basic_predicate)] == 1 :
                        continue

                    F, field = basic_predicate
                    field_predicate_1 = F(pair[0][field])
                    if not field_predicate_1 :
                        self._overlap[(pair, basic_predicate)] = -1 
                        break

                    field_predicate_2 = F(pair[1][field])

                    if set(field_predicate_1) & set(field_predicate_2) :
                        self._overlap[(pair, basic_predicate)] = 1 
                    else:
                        self._overlap[(pair, basic_predicate)] = -1
                        break
                    
                else:
                    coverage[predicate].append(pair)


        return coverage
            

    def createPredicateSet(self, disjunctive):

        # The set of simple predicates
        predicate_set = list(product(self.predicate_functions,
                                     self.fields))

        if disjunctive:
            disjunctive_predicates = list(combinations(predicate_set, 2))

            # filter out disjunctive predicates that operate on same
            # field
            disjunctive_predicates = [predicate for predicate
                                      in disjunctive_predicates
                                      if predicate[0][1]
                                      != predicate[1][1]]

            predicate_set = [(predicate, ) for predicate in
                             predicate_set]
            predicate_set.extend(disjunctive_predicates)
        else:

            predicate_set = [(predicate, ) for predicate in
                             predicate_set]

        return predicate_set

    def filterOutIndistinctPairs(self,
                                 expected_dupe_cover,
                                 found_distinct,
                                 training_distinct,
                                 ):


        # We don't want to penalize a blocker if it puts distinct
        # pairs together that look like they could be duplicates.
        predicate_count = defaultdict(int)
        for pair in chain(*found_distinct.values()):
            predicate_count[pair] += 1

        training_distinct = [pair for pair in training_distinct
                             if predicate_count[pair]
                             < expected_dupe_cover]

        return self.predicateCoverage(training_distinct)


    def findOptimumBlocking(self,
                            training_dupes,
                            predicate_set,
                            found_dupes,
                            found_distinct,
                            ):

        # Greedily find the predicates that, at each step, covers the
        # most duplicates and covers the least distinct pairs, due to
        # Chvatal, 1979
        final_predicate_set = []
        n_training_dupes = len(training_dupes)
        print 'Uncovered dupes'
        print n_training_dupes
        while n_training_dupes >= self.epsilon:

            optimumCover = 0
            bestPredicate = None
            for predicate in predicate_set:
                try:
                    cover = (len(found_dupes[predicate])
                             / float(len(found_distinct[predicate]))
                             )
                except ZeroDivisionError:
                    cover = len(found_dupes[predicate])

                if cover > optimumCover:
                    optimumCover = cover
                    bestPredicate = predicate

            if not bestPredicate:
                print 'Ran out of predicates'
                break

            predicate_set.remove(bestPredicate)
            n_training_dupes -= len(found_dupes[bestPredicate])
            [training_dupes.remove(pair) for pair in
             found_dupes[bestPredicate]]
            found_dupes = self.predicateCoverage(training_dupes)

            final_predicate_set.append(bestPredicate)

        return final_predicate_set
