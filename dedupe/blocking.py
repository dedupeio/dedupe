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
    def __init__(self, predicates):
        self.predicates = predicates

        self.tfidf_fields = set([])
        self.tfidf_predicates = set([])

        for predicate in predicates:
            for pred, field in predicate :
                if pred.__class__ is tfidf.TfidfPredicate :
                    self.tfidf_predicates.add((pred, field))
                    self.tfidf_fields.add(field)


    def __call__(self, instance) :
        record_id, record = instance 

        record_keys = []
        for predicate in self.predicates:
            predicate_keys = []
            for F, field in predicate :
                pred_id = F.__name__ + field
                if isinstance(F, types.FunctionType) :
                    record_field = record[field].strip().lower()
                    predicate_keys.append([str(key) + pred_id
                                           for key in F(record_field)])
                elif F.__class__ is tfidf.TfidfPredicate :
                    center = self.canopies[pred_id][record_id]
                    if center is not None :
                        key = str(center) + pred_id
                        predicate_keys.append((key,))
                    else:
                        continue

            record_keys.extend(product(*predicate_keys))

        #return record_keys
        return set([str(key) for key in record_keys])

    
    def invertIndex(self, data_d) :
        if not self.tfidf_fields:
            return None

        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.token_vector = defaultdict(dict)
        self.corpus_ids = set([])

        for record_id, record in data_d :
            self.corpus_ids.add(record_id) # candidate for removal
            for field in self.tfidf_fields :
                tokens = record[field].lower().replace(",", "").split()
                tokens = [(token, tokens.count(token)) for token in set(tokens)]
                for token, _ in tokens:
                    self.inverted_index[field][token].append(record_id)
                
                self.token_vector[field][record_id] = tokens

        # ignore stop words in TFIDF canopy creation
        num_docs = len(self.token_vector[field])

        stop_word_threshold = max(num_docs * 0.025, 1000)
        print "Stop word threshold: ", stop_word_threshold

        num_docs_log = math.log(num_docs + 0.5)
        singleton_idf = num_docs_log - math.log(1.0 + 0.5)

        for field in self.inverted_index:
            for token, occurrences in self.inverted_index[field].iteritems() :
                n_occurrences = len(occurrences)
                if n_occurrences < 2 :
                    idf = singleton_idf
                    occurrences = []
                else :
                    idf = num_docs_log - math.log(n_occurrences + 0.5)
                    if n_occurrences > stop_word_threshold :
                        occurrences = []
                        print "Stop word: ", field, token, n_occurrences

                self.inverted_index[field][token] = {'idf' : idf,
                                                     'occurrences' : occurrences}

        for field in self.token_vector:
            inverted_index = self.inverted_index[field]
            for record_id, tokens in self.token_vector[field].iteritems():
                norm = math.sqrt(sum((inverted_index[token]['idf'] * count)**2
                                     for token, count in tokens))
                self.token_vector[field][record_id] = (dict(tokens), norm)

    def tfIdfBlocks(self, data) :
        self.invertIndex(data)
        self.canopies = {}
        
        print 'creating TF/IDF canopies'

        num_thresholds = len(self.tfidf_predicates)

        for i, (threshold, field) in enumerate(self.tfidf_predicates, 1) :
            print (str(i) + "/" + str(num_thresholds)), threshold, field
            canopy = self.createCanopies(field, threshold)
            self.canopies[threshold.__name__ + field] = canopy


    def createCanopies(self, field, threshold) :
      """
      A function that returns 
      a field value of a record with a particular doc_id, doc_id
      is the only argument that must be accepted by select_function
      """

      canopies = defaultdict(lambda:None)
      seen_set = set([])
      corpus_ids = self.corpus_ids.copy()

      token_vectors = self.token_vector[field]
      while corpus_ids :
        center_id = corpus_ids.pop()
        canopies[center_id] = center_id

        doc_id = center_id
        center_vector, center_norm = token_vectors[center_id]

        seen_set.add(center_id)

        if not center_norm :
            continue    

        # initialize the potential block with center
        candidate_set = set([])

        for token in center_vector :
          candidate_set.update(self.inverted_index[field][token]['occurrences'])

        candidate_set = candidate_set - seen_set
        for doc_id in candidate_set :
          candidate_vector, candidate_norm = token_vectors[doc_id]
          if not candidate_norm :
            continue

          common_tokens = set(center_vector.keys()).intersection(candidate_vector.keys())

          dot_product = 0 
          for token in common_tokens :
            token_idf = self.inverted_index[field][token]['idf']
            dot_product += (center_vector[token] * token_idf) * (candidate_vector[token] * token_idf)

          similarity = dot_product / (center_norm * candidate_norm)

          if similarity > threshold :
            canopies[doc_id] = center_id
            seen_set.add(doc_id)
            corpus_ids.remove(doc_id)



      return canopies



def blockTraining(training_pairs,
                  predicate_functions,
                  data_model,
                  tfidf_thresholds = [],
                  df_index = {},
                  eta = .1,
                  epsilon = .1) : 


    (training_dupes,
     training_distinct,
     predicate_set,
     _overlap) =  _initializeTraining(training_pairs,
                                      data_model,
                                      predicate_functions,
                                      tfidf_thresholds,
                                      df_index)

        
    coverage_threshold = eta * len(training_distinct)


    n_predicates = len(predicate_set)


    found_dupes, _overlap = predicateCoverage(predicate_set,
                                              training_dupes,
                                              _overlap)


    # Only consider predicates that cover at least one duplicate pair
    predicate_set = found_dupes.keys()
    
    (found_distinct,
     distinct_blocks,
     _overlap) = predicateCoverage(predicate_set, 
                                   training_distinct,
                                   _overlap,
                                   return_blocks = True)


    # We want to throw away the predicates that puts together too
    # many distinct pairs
    for pred, blocking in distinct_blocks.iteritems() :
        if any(len(record_ids) >= coverage_threshold
               for record_ids in blocking.values()) :
            predicate_set.remove(pred)

    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              found_dupes,
                                              found_distinct,
                                              epsilon,
                                              _overlap)

    print 'Final predicate set:'
    print final_predicate_set

    if final_predicate_set:
        return final_predicate_set
    else:
        print 'No predicate found!'
        raise


def _initializeTraining(training_pairs,
                        data_model,
                        predicate_functions,
                        tfidf_thresholds,
                        df_index) :

    training_dupes = (training_pairs[1])[:]
    training_distinct = (training_pairs[0])[:]

    fields = [field for field in data_model['fields']
              if data_model['fields'][field]['type']
              != 'Interaction']

    predicate_functions = list(product(predicate_functions,
                                       fields))

    tfidf_predicates = [tfidf.TfidfPredicate(threshold)
                        for threshold in tfidf_thresholds]
    tfidf_predicates = list(product(tfidf_predicates, fields))

    predicate_set = disjunctivePredicates(predicate_functions + tfidf_predicates)

    if tfidf_predicates :
        _overlap = canopyOverlap(tfidf_predicates,
                                 training_dupes + training_distinct,
                                 df_index) 
    else:
        _overlap = defaultdict(lambda:None)

    return (training_dupes, training_distinct, predicate_set, _overlap)



def predicateCoverage(predicate_set, pairs, _overlap, return_blocks=False):
    coverage = defaultdict(list)
    blocks = defaultdict(lambda: defaultdict(set))
    for pair in pairs:
        for predicate in predicate_set:
            for basic_predicate in predicate :
                if _overlap[(pair, basic_predicate)] == -1 :
                    break
                if _overlap[(pair, basic_predicate)] == 1 :
                    continue

                F, field = basic_predicate
                field_predicate_1 = F(pair[0][field])
                if not field_predicate_1 :
                    _overlap[(pair, basic_predicate)] = -1 
                    break

                field_predicate_2 = F(pair[1][field])

                if set(field_predicate_1) & set(field_predicate_2) :
                    _overlap[(pair, basic_predicate)] = 1 
                else:
                    _overlap[(pair, basic_predicate)] = -1
                    break

            else:
                coverage[predicate].append(pair)
                if return_blocks: 
                    blocks[predicate][(field_predicate_1,
                                       field_predicate_2)].update(pair)

    if return_blocks :
        return coverage, blocks, _overlap

    else :
        return coverage, _overlap


def findOptimumBlocking(training_dupes,
                        predicate_set,
                        found_dupes,
                        found_distinct,
                        epsilon,
                        _overlap,
                        ):

    # Greedily find the predicates that, at each step, covers the
    # most duplicates and covers the least distinct pairs, due to
    # Chvatal, 1979
    final_predicate_set = []
    n_training_dupes = len(training_dupes)
    print 'Uncovered dupes: ', n_training_dupes
    while n_training_dupes >= epsilon:

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
        found_dupes, _overlap = predicateCoverage(predicate_set,
                                                  training_dupes,
                                                  _overlap)

        final_predicate_set.append(bestPredicate)

    return final_predicate_set

            

def disjunctivePredicates(predicate_set) :

    disjunctive_predicates = list(combinations(predicate_set, 2))

    # filter out disjunctive predicates that operate on same field
    disjunctive_predicates = [predicate for predicate
                              in disjunctive_predicates
                              if predicate[0][1]
                              != predicate[1][1]]

    predicate_set = [(predicate, ) for predicate in
                     predicate_set]
    predicate_set.extend(disjunctive_predicates)

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



def blockingIndex(data_d, blocker):

    blocks = defaultdict(list)

    blocker.tfIdfBlocks(data_d.iteritems())

    for record_id, record in data_d.iteritems() :
        for key in blocker((record_id, record)):
            blocks[key].append((record_id, record))

    for i, key in enumerate(blocks) :
        yield blocks[key]

def semiSupervisedNonDuplicates(data_sample, data_model, 
                                nonduplicate_confidence_threshold=.7,
                                sample_size = 2000):



    if len(data_sample) <= sample_size :
        return data_sample

    confident_distinct_pairs = []
    n_distinct_pairs = 0
    for pair in data_sample :

        pair_distance = core.recordDistances([pair], data_model)
        score = core.scorePairs(pair_distance, data_model)


        if score < (1 - nonduplicate_confidence_threshold):
            key_pair, value_pair = zip(*pair)
            confident_distinct_pairs.append(value_pair)
            n_distinct_pairs += 1
            if n_distinct_pairs == sample_size :
                return confident_distinct_pairs


def canopyOverlap(tfidf_predicates,
                  record_pairs,
                  df_index) :

    overlap = defaultdict(lambda:None)
    for threshold, field in tfidf_predicates :
        coverage =  tfidf.coverage(threshold, 
                                   field, 
                                   record_pairs,
                                   df_index)

        for pair, value in coverage.iteritems():
            overlap[(pair, (threshold, field))] = value

    return overlap
