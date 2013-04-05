#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import product, chain, combinations
import types
import math
import logging


import dedupe.tfidf as tfidf



class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, predicates):
        self.predicates = predicates

        self.tfidf_fields = set([])
        self.tfidf_predicates = set([])

        for predicate in predicates:
            for (pred, field) in predicate:
                if pred.__class__ is tfidf.TfidfPredicate:
                    self.tfidf_predicates.add((pred, field))
                    self.tfidf_fields.add(field)

        self.canopies = None

    def __call__(self, instance):
        (record_id, record) = instance

        record_keys = []
        for predicate in self.predicates:
            predicate_keys = []
            for (F, field) in predicate:
                pred_id = F.__name__ + field
                if isinstance(F, types.FunctionType):
                    record_field = record[field].strip().lower()
                    block_keys = [str(key) + pred_id for key in F(record_field)]
                    predicate_keys.append(block_keys)
                elif F.__class__ is tfidf.TfidfPredicate:
                    center = self.canopies[pred_id][record_id]
                    if center is not None:
                        key = str(center) + pred_id
                        predicate_keys.append((key, ))
                    else:
                        continue

            record_keys.extend(product(*predicate_keys))

        return set([str(key) for key in record_keys])

    def tfIdfBlocks(self, data, df_index=None):
        '''Creates TF/IDF canopy of a given set of data'''
        if self.tfidf_fields:

            (inverted_index, token_vector, corpus_ids) = \
                invertIndex(data, self.tfidf_fields, df_index)

        self.canopies = {}

        logging.info('creating TF/IDF canopies')

        num_thresholds = len(self.tfidf_predicates)

        for (i, (threshold, field)) in enumerate(self.tfidf_predicates, 1):
            logging.info('%(i)i/%(num_thresholds)i field %(threshold)2.2f %(field)s',
                         {'i': i, 
                          'num_thresholds': num_thresholds, 
                          'threshold': threshold, 
                          'field': field})

            canopy = createCanopies(field, threshold, corpus_ids,
                                    token_vector, inverted_index)
            self.canopies[threshold.__name__ + field] = canopy


# TODO: Split this into subfunctions

def invertIndex(data_d, tfidf_fields, df_index=None):

    inverted_index = defaultdict(lambda : defaultdict(list))
    token_vector = defaultdict(dict)
    corpus_ids = set([])

    for (record_id, record) in data_d:
        corpus_ids.add(record_id)  # candidate for removal
        for field in tfidf_fields:
            tokens = record[field].lower().replace(',', '').split()
            tokens = [(token, tokens.count(token)) for token in set(tokens)]
            for (token, _) in tokens:
                inverted_index[field][token].append(record_id)

            token_vector[field][record_id] = tokens

    # ignore stop words in TFIDF canopy creation

    num_docs = len(token_vector.values()[0])

    stop_word_threshold = max(num_docs * 0.025, 100)
    logging.info('Stop word threshold: %(stop_thresh)d',
                 {'stop_thresh' :stop_word_threshold})

    num_docs_log = math.log(num_docs + 0.5)
    singleton_idf = num_docs_log - math.log(1.0 + 0.5)

    if df_index:
        for field in inverted_index:
            for (token, occurrences) in \
                inverted_index[field].iteritems():
                inverted_index[field][token] = {'idf': df_index[token],
                                                'occurrences': set(occurrences)}
    else:

        for field in inverted_index:
            for (token, occurrences) in \
                inverted_index[field].iteritems():
                n_occurrences = len(occurrences)
                if n_occurrences < 2:
                    idf = singleton_idf
                    occurrences = []
                else:
                    idf = num_docs_log - math.log(n_occurrences + 0.5)
                    if n_occurrences > stop_word_threshold:
                        occurrences = []
                        logging.info('Stop word: %(field)s, %(token)s, %(occurences)d',
                                     {'field' : field,
                                      'token' : token,
                                      'occurences' : n_occurrences})

                inverted_index[field][token] = {'idf': idf, 
                                                'occurrences': set(occurrences)}

    for field in token_vector:
        field_inverted_index = inverted_index[field]
        for (record_id, tokens) in token_vector[field].iteritems():
            norm = math.sqrt(sum((field_inverted_index[token]['idf'] * count)**2 
                                  for (token, count) in tokens))
            if norm > 0 :
                token_vector[field][record_id] = (dict(tokens), norm)
            else :
                token_vector[field][record_id] = ({}, 0)
    return (inverted_index, token_vector, corpus_ids)

def createCanopies(field,
                   threshold,
                   corpus_ids,
                   token_vector,
                   inverted_index):
    """
    A function that returns a field value of a record with a
    particular doc_id, doc_id is the only argument that must be
    accepted by select_function
    """

    canopies = defaultdict(lambda : None)
    seen_set = set([])
    corpus_ids = corpus_ids.copy()
    field_inverted_index = inverted_index[field]

    token_vectors = token_vector[field]
    while corpus_ids:
        center_id = corpus_ids.pop()
        canopies[center_id] = center_id

        doc_id = center_id
        (center_vector, center_norm) = token_vectors[center_id]

        seen_set.add(center_id)

        if not center_norm:
            continue

        # initialize the potential block with center
        center_tokens = set(token for token
                            in center_vector.keys()
                            if field_inverted_index[token]['idf'] > 0)

        candidate_set = set.union(*(field_inverted_index[token]['occurrences']
                                    for token in center_tokens))

        candidate_set = candidate_set - seen_set


        token_idfs = dict([(token, field_inverted_index[token]['idf']**2)
                           for token in center_tokens])
        center_threshold = threshold * center_norm

        for doc_id in candidate_set:
            (candidate_vector, candidate_norm) = token_vectors[doc_id]

            common_tokens = center_tokens.intersection(candidate_vector.keys())

            cosine_similarity = sum((center_vector[token]
                                     * candidate_vector[token]
                                     * token_idfs[token])
                                    for token in common_tokens)/candidate_norm
            

            if cosine_similarity > center_threshold :
                canopies[doc_id] = center_id
                seen_set.add(doc_id)
                corpus_ids.remove(doc_id)

    return canopies

def blockTraining(training_pairs,
                  predicate_functions,
                  fields,
                  tfidf_thresholds=None,
                  eta=.1,
                  epsilon=.1):
    '''Takes in a set of training pairs and predicates and tries to find a good set of blocking rules.'''
    
    (training_dupes, 
     training_distinct, 
     predicate_set, 
     _overlap) = _initializeTraining(training_pairs, 
                                     fields,
                                     predicate_functions, 
                                     tfidf_thresholds)


    coverage_threshold = eta * len(training_distinct)
    logging.info("coverage threshold: %s", coverage_threshold)

    n_predicates = len(predicate_set)

    (found_dupes, _overlap) = predicateCoverage(predicate_set, training_dupes, _overlap)

    # Only consider predicates that cover at least one duplicate pair

    predicate_set = found_dupes.keys()

    (found_distinct, 
     blocks, 
     _overlap) = predicateCoverage(predicate_set, 
                                   training_distinct, 
                                   _overlap,
                                   return_blocks=True)

    # We want to throw away the predicates that puts together too
    # many distinct pairs

    logging.info("Before removing liberal predicates, %s predicates",
                 len(predicate_set))

    for (pred, blocks) in blocks.iteritems():
        if any(len(block) >= 0 for block in blocks if block):
            predicate_set.remove(pred)

    logging.info("After removing liberal predicates, %s predicates",
                 len(predicate_set))


    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              found_dupes,
                                              found_distinct,
                                              epsilon,
                                              _overlap)

    logging.info('Final predicate set:')
    logging.info(final_predicate_set)

    if final_predicate_set:
        return final_predicate_set
    else:
        raise ValueError('No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data')


def _initializeTraining(training_pairs,
                        fields,
                        predicate_functions,
                        tfidf_thresholds) :


    training_dupes = (training_pairs[1])[:]
    training_distinct = (training_pairs[0])[:]

    predicate_functions = list(product(predicate_functions, fields))

    tfidf_predicates = [tfidf.TfidfPredicate(threshold)
                        for threshold in tfidf_thresholds]
    tfidf_predicates = list(product(tfidf_predicates, fields))

    predicate_set = disjunctivePredicates(predicate_functions
            + tfidf_predicates)

    if tfidf_predicates:
        _overlap = canopyOverlap(predicate_set, 
                                 training_dupes + training_distinct)

    else:
        _overlap = Overlap()

    _overlap = simplePredicateOverlap(predicate_functions,
                                      training_dupes + training_distinct,
                                      _overlap)

    return (training_dupes, training_distinct, predicate_set, _overlap)

    
        

#@profile
def predicateCoverage(predicate_set,
                      pairs,
                      _overlap,
                      return_blocks=False):
    coverage = defaultdict(list)
    predicate_blocks = {}

    pairs = set(pairs)

    _overlapping = defaultdict(set)
    for basic_predicate, covered_pairs in _overlap.overlapping.iteritems() :
        _overlapping[basic_predicate] = pairs.intersection(covered_pairs)

    for predicate in predicate_set :
        covered_pairs = set.intersection(*(_overlapping[basic_predicate]
                                           for basic_predicate
                                           in predicate))
        coverage[predicate] = covered_pairs

    if return_blocks :
        _blocks = defaultdict(lambda : defaultdict(set))
        for basic_predicate in _overlap.blocks :
            for block_key, block_group in _overlap.blocks[basic_predicate].iteritems() :

                block_group = pairs.intersection(block_group)
                if block_group :
                    _blocks[basic_predicate][block_key] = block_group
        for predicate in predicate_set :
            block_groups = product(*(_blocks[basic_predicate].values()
                                     for basic_predicate
                                     in predicate))


            block_groups = (set.intersection(*block_group)
                            for block_group in block_groups)
            predicate_blocks[predicate] = block_groups


    if return_blocks:
        return (coverage, predicate_blocks, _overlap)
    else:
        return (coverage, _overlap)




def findOptimumBlocking(training_dupes,
                        predicate_set,
                        found_dupes,
                        found_distinct,
                        epsilon,
                        _overlap):

    # Greedily find the predicates that, at each step, covers the
    # most duplicates and covers the least distinct pairs, due to
    # Chvatal, 1979

    final_predicate_set = []
    n_training_dupes = len(training_dupes)
    logging.info('Uncovered dupes: %(n_dupes)d',
                 {'n_dupes' : n_training_dupes})
    while n_training_dupes >= epsilon:

        optimum_cover = 0
        best_predicate = None
        for predicate in predicate_set:
            cover = len(found_dupes[predicate]) / (float(len(found_distinct[predicate])) + 0.5)
            if cover > optimum_cover and cover > 1:
                optimum_cover = cover
                best_predicate = predicate

                logging.info(best_predicate)
                logging.info('cover: %(cover)d, found_dupes: %(found_dupes)d, '
                             'found_distinct: %(found_distinct)d',
                             {'cover' : cover,
                              'found_dupes' : len(found_dupes[best_predicate]),
                              'found_distinct' : len(found_distinct[best_predicate])})

        if not best_predicate:
            logging.warning('Ran out of predicates')
            break

        predicate_set.remove(best_predicate)
        n_training_dupes -= len(found_dupes[best_predicate])
        
        [training_dupes.remove(pair) for pair in found_dupes[best_predicate]]
        
        (found_dupes, _overlap) = predicateCoverage(predicate_set, training_dupes, _overlap)

        final_predicate_set.append(best_predicate)

    return final_predicate_set


def disjunctivePredicates(predicate_set):

    disjunctive_predicates = list(combinations(predicate_set, 2))

    # filter out disjunctive predicates that operate on same field

    disjunctive_predicates = [predicate for predicate in disjunctive_predicates 
                              if predicate[0][1] != predicate[1][1]]

    predicate_set = [(predicate, ) for predicate in predicate_set]
    predicate_set.extend(disjunctive_predicates)

    return predicate_set

class Overlap() :
    def __init__(self) :
        self.overlapping = defaultdict(set)
        self.blocks = defaultdict(lambda : defaultdict(set))
    
def simplePredicateOverlap(predicates,
                           pairs,
                           overlap) :

    for basic_predicate in predicates :
        (F, field) = basic_predicate        
        for pair in pairs :
            field_predicate_1 = F(pair[0][field])

            if field_predicate_1:
                field_predicate_2 = F(pair[1][field])

                if field_predicate_2 :
                    field_preds = set(field_predicate_2) & set(field_predicate_1)
                    if field_preds :
                        overlap.overlapping[basic_predicate].add(pair)

                    for field_pred in field_preds :
                        overlap.blocks[basic_predicate][field_pred].add(pair)

    return overlap



def canopyOverlap(tfidf_predicates, record_pairs) :

    overlap = Overlap()

    docs = list(set(chain(*record_pairs)))
    enumerated_docs = zip(docs, docs)

    blocker = Blocker(tfidf_predicates)
    blocker.tfIdfBlocks(enumerated_docs)

    for (threshold, field) in blocker.tfidf_predicates:
        canopy = blocker.canopies[threshold.__name__ + field]
        for record_1, record_2 in record_pairs :
            if canopy[record_1] == canopy[record_2]:
                overlap.overlapping[(threshold, field)].add((record_1, record_2))
                overlap.blocks[(threshold, field)][canopy[record_1]].add((record_1, record_2))

    return overlap
