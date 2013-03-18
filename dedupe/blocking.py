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

    stop_word_threshold = max(num_docs * 0.025, 1000)
    logging.info('Stop word threshold: %(stop_thresh)d',
                 {'stop_thresh' :stop_word_threshold})

    num_docs_log = math.log(num_docs + 0.5)
    singleton_idf = num_docs_log - math.log(1.0 + 0.5)

    if df_index:
        for field in inverted_index:
            for (token, occurrences) in \
                inverted_index[field].iteritems():
                inverted_index[field][token] = {'idf': df_index[token],
                                                'occurrences': occurrences}
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
                                                'occurrences': occurrences}

    for field in token_vector:
        field_inverted_index = inverted_index[field]
        for (record_id, tokens) in token_vector[field].iteritems():
            norm = math.sqrt(sum((field_inverted_index[token]['idf'] * count)**2 
                                  for (token, count) in tokens))
            token_vector[field][record_id] = (dict(tokens), norm)

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

        candidate_set = set([])

        for token in center_vector:
            candidate_set.update(inverted_index[field][token]['occurrences'])

        candidate_set = candidate_set - seen_set
        for doc_id in candidate_set:
            (candidate_vector, candidate_norm) = token_vectors[doc_id]
            if not candidate_norm:
                continue

            common_tokens = set(center_vector.keys()).intersection(candidate_vector.keys())

            dot_product = 0
            for token in common_tokens:
                token_idf = inverted_index[field][token]['idf']
                dot_product += center_vector[token] * token_idf * (candidate_vector[token] * token_idf)

            cosine_similarity = dot_product / (center_norm * candidate_norm)

            if cosine_similarity > threshold:
                canopies[doc_id] = center_id
                seen_set.add(doc_id)
                corpus_ids.remove(doc_id)

    return canopies


def blockTraining(training_pairs,
                  predicate_set,
                  df_index=None,
                  eta=.1,
                  epsilon=.1):
    '''Takes in a set of training pairs and predicates and tries to find a good set of blocking rules.'''

    (training_dupes, 
     training_distinct, 
     _overlap) = _initializeTraining(training_pairs,
                                     predicate_set,
                                     df_index)

    coverage_threshold = eta * len(training_distinct)

    n_predicates = len(predicate_set)

    (found_dupes, _overlap) = predicateCoverage(predicate_set, training_dupes, _overlap)

    # Only consider predicates that cover at least one duplicate pair

    predicate_set = found_dupes.keys()

    (found_distinct, 
     distinct_blocks, 
     _overlap) = predicateCoverage(predicate_set, 
                                   training_distinct, 
                                   _overlap,
                                   return_blocks=True)

    # We want to throw away the predicates that puts together too
    # many distinct pairs

    for (pred, blocking) in distinct_blocks.iteritems():
        if any(len(record_ids) >= coverage_threshold for record_ids in blocking.values()):
            predicate_set.remove(pred)

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
        raise ValueError('No predicate found!')


def _initializeTraining(training_pairs,
                        predicate_set,
                        df_index):

    training_dupes = (training_pairs[1])[:]
    training_distinct = (training_pairs[0])[:]

    _overlap = canopyOverlap(predicate_set, 
                             training_dupes + training_distinct, 
                             df_index)

    return (training_dupes, training_distinct, _overlap)


def predicateCoverage(predicate_set,
                      pairs,
                      _overlap,
                      return_blocks=False):
    coverage = defaultdict(list)
    blocks = defaultdict(lambda : defaultdict(set))
    for pair in pairs:
        for predicate in predicate_set:
            for basic_predicate in predicate:
                if _overlap[(pair, basic_predicate)] == -1:
                    break
                if _overlap[(pair, basic_predicate)] == 1:
                    continue

                (F, field) = basic_predicate
                field_predicate_1 = F(pair[0][field])
                if not field_predicate_1:
                    _overlap[(pair, basic_predicate)] = -1
                    break

                field_predicate_2 = F(pair[1][field])

                if set(field_predicate_1) & set(field_predicate_2):
                    _overlap[(pair, basic_predicate)] = 1
                else:
                    _overlap[(pair, basic_predicate)] = -1
                    break
            else:

                coverage[predicate].append(pair)
                if return_blocks:
                    blocks[predicate][(field_predicate_1, field_predicate_2)].update(pair)

    if return_blocks:
        return (coverage, blocks, _overlap)
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



def canopyOverlap(tfidf_predicates, record_pairs, df_index):

    overlap = defaultdict(lambda : None)

    docs = list(set(instance for pair in record_pairs for instance in pair))
    enumerated_docs = zip(xrange(len(docs)), docs)

    id_lookup = dict(zip(docs, xrange(len(docs))))

    blocker = Blocker(tfidf_predicates)
    blocker.tfIdfBlocks(enumerated_docs, df_index)

    for (threshold, field) in blocker.tfidf_predicates:
        canopy_group = threshold.__name__ + field
        for (record_1, record_2) in record_pairs:
            id_1 = id_lookup[record_1]
            id_2 = id_lookup[record_2]

            canopy_id_1 = blocker.canopies[canopy_group][id_1]
            canopy_id_2 = blocker.canopies[canopy_group][id_2]

            if canopy_id_1 == canopy_id_2:
                overlap[((record_1, record_2), (threshold, field))] = 1
            else:
                overlap[((record_1, record_2), (threshold, field))] = -1

    return overlap
