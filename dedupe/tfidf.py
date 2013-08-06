#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import logging
import re

words = re.compile("[\w']+")

class TfidfPredicate(float):
    def __new__(self, threshold):
        return float.__new__(self, threshold)

    def __init__(self, threshold):
        self.__name__ = 'TF-IDF:' + str(threshold)

def invertIndex(data, tfidf_fields, constrained_matching=False, df_index=None):

    inverted_index = defaultdict(lambda : defaultdict(list))
    constrained_inverted_index = defaultdict(lambda : defaultdict(list))
    token_vector = defaultdict(dict)
    corpus_ids = set([])

    for (record_id, record) in data:
        if constrained_matching:
            if record['dataset'] == 0:
                corpus_ids.add(record_id)  # candidate for removal
        else:
            corpus_ids.add(record_id)  # candidate for removal
        for field in tfidf_fields:
            tokens = words.findall(record[field].lower())
            tokens = [(token, tokens.count(token))
                      for token in set(tokens)
                      if token]
            for (token, _) in tokens:
                inverted_index[field][token].append(record_id)
            if constrained_matching :
                for (token, _) in tokens:
                    constrained_inverted_index[field][token].append(record_id)

            token_vector[field][record_id] = tokens

    # ignore stop words in TFIDF canopy creation

    num_docs = len(token_vector.values()[0])

    stop_word_threshold = max(num_docs * 0.025, 500)
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
                if constrained_matching :
                    inverted_index[field][token] = {'idf': idf, 
                                                    'occurrences': set(constrained_inverted_index[field][token])}
                else :
                    inverted_index[field][token] = {'idf': idf, 
                                                    'occurrences': set(occurences)}

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
                   data,
                   threshold,
                   corpus_ids,
                   token_vector,
                   inverted_index,
                   constrained_matching=False):
    """
    A function that returns a field value of a record with a
    particular doc_id, doc_id is the only argument that must be
    accepted by select_function
    """

    canopies = defaultdict(lambda : None)
    seen_set = set([])
    corpus_ids = corpus_ids.copy()
    field_inverted_index = inverted_index[field]
    data_d = dict(data)

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

        if constrained_matching:
            candidate_set = set((doc_id for token in center_tokens 
                                        for doc_id in field_inverted_index[token]['occurrences']))

        else:
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
                if not constrained_matching:
                    corpus_ids.remove(doc_id)

    return canopies
