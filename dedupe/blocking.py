#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import product, chain, combinations
from math import sqrt, log
import core
import tfidf
from random import sample, random, choice, shuffle
import types
import math

class Blocker: 
    def __init__(self, predicates, df_index):
        self.predicates = predicates
        self.df_index = df_index
        self.simple_predicates = []
        self.tfidf_thresholds = set([])
        self.mixed_predicates = []
        self.tfidf_fields = set([])
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.shim_tfidf_thresholds = []
        self.token_vector = defaultdict(dict)

        self.corpus_ids = set([])

        for predicate in predicates:
            for pred in predicate :
                if pred[0].__class__ is tfidf.TfidfPredicate :
                    self.tfidf_thresholds.add(pred)


    def __call__(self, instance) :
        record_id, record = instance 
        keys = {}
        record_keys = []
        for predicate in self.predicates:
            predicate_keys = []
            for F, field in predicate :
                pred_id = F.__name__ + field
                if isinstance(F, types.FunctionType) :
                    predicate_keys.append([str(key) + pred_id
                                           for key in F(record[field])])
                elif F.__class__ is tfidf.TfidfPredicate :
                    #print self.canopies
                    #raise
                    center = self.canopies[pred_id][record_id]
                    if center is not None :
                        key = str(center) + pred_id
                        predicate_keys.append((key,))

            record_keys.extend(product(*predicate_keys))

        return record_keys

    
    def invertIndex(self, data_d) :
        for record_id, record in data_d :
            for _, field in self.tfidf_thresholds :
                tokens = record[str(field)].lower().split()
                tokens = [(token, tokens.count(token)) for token in set(tokens)]
                for token, _ in tokens:
                    self.inverted_index[field][token].append(record_id)
                
                self.corpus_ids.add(record_id) # candidate for removal
                self.token_vector[field][record_id] = tokens

        # ignore stop words in TFIDF canopy creation
        num_docs = len(self.token_vector[field])
        stop_word_threshold = num_docs * 0.05

        singleton_idf = math.log((num_docs + 0.5) / (1.0 + 0.5))
        for field in self.inverted_index:
            for token, occurrences in self.inverted_index[field].iteritems() :
                n_occurrences = len(occurrences)
                if n_occurrences == 1 :
                    self.inverted_index[field][token] = {'idf' : singleton_idf,
                                                         'occurrences' : []}
                else : 
                    idf = math.log((num_docs + 0.5) / (n_occurrences + 0.5))
                    if n_occurrences > stop_word_threshold :
                        occurrences = []
                        print token
                    self.inverted_index[field][token] = {'idf' : idf,
                                                         'occurrences' : occurrences}

        for field in self.token_vector:
            for record_id, tokens in self.token_vector[field].iteritems():
                norm = 0.0
                for token, count in tokens:
                    token_idf = self.inverted_index[field][token]['idf']
                    norm += (count * token_idf) * (count * token_idf)
                norm = math.sqrt(norm)
                self.token_vector[field][record_id] = (dict(tokens), norm)

    def createCanopies(self, field, threshold) :
      """
      A function that returns 
      a field value of a record with a particular doc_id, doc_id
      is the only argument that must be accepted by select_function
      """

      blocked_data = defaultdict(lambda:None)
      seen_set = set([])
      corpus_ids = self.corpus_ids.copy()

      token_vectors = self.token_vector[field]
      while corpus_ids :
        center_id = corpus_ids.pop()
        blocked_data[center_id] = center_id

        doc_id = center_id
        center_vector, center_norm = token_vectors[center_id]

        seen_set.add(center_id)

        if not center_norm :
            continue    
        #print "center_id", center_id
        # print doc_id, center
        # if not center :
        #   continue

        # initialize the potential block with center
        candidate_set = set([])

        for token in center_vector :
          candidate_set.update(self.inverted_index[field][token]['occurrences'])

        # print candidate_set
        candidate_set = candidate_set - seen_set
        for doc_id in candidate_set :
          #print doc_id, candidate_field
          candidate_vector, candidate_norm = token_vectors[doc_id]
          if not candidate_norm :
            continue

          common_tokens = set(center_vector.keys()).intersection(candidate_vector.keys())

          dot_product = 0 
          for token in common_tokens :
            token_idf = self.inverted_index[field][token]['idf']
            dot_product += (center_vector[token] * token_idf) * (candidate_vector[token] * token_idf)

          similarity = dot_product / (center_norm * candidate_norm)

          if similarity > threshold.threshold :
            blocked_data[doc_id] = center_id
            seen_set.add(doc_id)
            corpus_ids.remove(doc_id)

            ## print threshold.threshold
            ## print center_id, center, center_dict
            ## print doc_id, candidate_dict
            ## print similarity


      return blocked_data


def blockingIndex(data_d, blocker):

    blocks = defaultdict(list)

    blocker.invertIndex(data_d.iteritems())

    blocker.canopies = {}
    for threshold, field in blocker.tfidf_thresholds :
        selector = lambda record_id : data_d[record_id][field]    
        # print field
        canopy = blocker.createCanopies(selector, field, threshold)
        # print blocks
        blocker.canopies[threshold.__name__ + field] = canopy

    for record_id, record in data_d.iteritems() :
        for key in blocker((record_id, record)):
            blocks[key].append((record_id, record))
    
    return blocks


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
                 epsilon=5,
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
        print 'coverage threshold:', self.coverage_threshold
                

    # Approximate learning of blocking following the ApproxRBSetCover from
    # page 102 of Bilenko
    def trainBlocking(self, disjunctive=True):
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



                    
    
        #print self.predicate_set
        #print self._overlap
        found_dupes = self.predicateCoverage(self.training_dupes)
        found_distinct, distinct_blocks = self.predicateCoverage(self.training_distinct,
                                                return_blocks = True)

        max_block_sizes = {}
        for pred, blocking in distinct_blocks.iteritems() :
            max_block_size = max(len(v) for v in blocking.values())
            #print max_block_size

            max_block_sizes[pred] = max_block_size
                

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
         in set(self.predicate_set).intersection(found_distinct.keys())
         if max_block_sizes[predicate] >= self.coverage_threshold]

        # Expected number of predicates that should cover a duplicate
        # pair
        expected_dupe_cover = sqrt(n_predicates
                                   / log(len(self.training_dupes)))

        #found_distinct = self.filterOutIndistinctPairs(expected_dupe_cover,
        #                                               found_distinct,
        #                                               self.training_distinct)

        final_predicate_set = self.findOptimumBlocking(self.training_dupes,
                                                       self.predicate_set,
                                                       found_dupes,
                                                       found_distinct)

        print 'Final predicate set:'
        print final_predicate_set

        if final_predicate_set:
            return final_predicate_set
        else:
            print 'No predicate found!'
            raise

    def predicateCoverage(self, pairs, return_blocks=False):
        coverage = defaultdict(list)
        blocks = defaultdict(lambda: defaultdict(set))
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
                    if return_blocks: 
                        blocks[predicate][(field_predicate_1,
                                           field_predicate_2)].update(pair)

        if return_blocks :
            return coverage, blocks

        else :
            return coverage
            

    def createPredicateSet(self, disjunctive):

        # The set of simple predicates
        predicate_set = list(product(self.predicate_functions,
                                     self.fields))


        predicate_set.extend(product(self.tfidf_thresholds,
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
        print 'Uncovered dupes: ', n_training_dupes
        while n_training_dupes >= self.epsilon:

            optimumCover = 0
            bestPredicate = None
            for predicate in predicate_set:
                cover = (len(found_dupes[predicate])
                         / (float(len(found_distinct[predicate]))
                            + 0.5)
                         )
                if cover > optimumCover and cover > 1 :
                    optimumCover = cover
                    bestPredicate = predicate

                    print bestPredicate
                    print 'cover:', cover, 'found_dupes:', len(found_dupes[bestPredicate]), 'found_distinct:', len(found_distinct[bestPredicate]) 

            if not bestPredicate:
                print 'WARNING: Ran out of predicates'
                break

            predicate_set.remove(bestPredicate)
            n_training_dupes -= len(found_dupes[bestPredicate])
            [training_dupes.remove(pair) for pair in
             found_dupes[bestPredicate]]
            found_dupes = self.predicateCoverage(training_dupes)

            final_predicate_set.append(bestPredicate)

        return final_predicate_set