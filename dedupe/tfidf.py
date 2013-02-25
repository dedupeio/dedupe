#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import math
import re
import core


class TfidfPredicate(float):
    def __new__(self, threshold):
        return float.__new__(self, threshold)

    def __init__(self, threshold):
        self.__name__ = 'TF-IDF:' + str(threshold)


def documentFrequency(corpus):
    num_docs = 0
    term_num_docs = collections.defaultdict(int)
    num_docs = len(corpus)
    stop_word_threshold = num_docs * 1
    stop_words = []

    for (doc_id, doc) in corpus.iteritems():
        tokens = getTokens(doc)
        for token in set(tokens):
            term_num_docs[token] += 1

    for (term, count) in term_num_docs.iteritems():
        if count < stop_word_threshold:
            term_num_docs[term] = math.log((num_docs + 0.5) / (float(count) + 0.5))
        else:
            term_num_docs[term] = 0
            stop_words.append(term)

    if stop_words:
        print 'stop words:', stop_words

    # term : num_docs_containing_term
    term_num_docs_default = collections.defaultdict(lambda : math.log((num_docs + 0.5) / 0.5))
    term_num_docs_default.update(term_num_docs)

    return term_num_docs_default


def getTokens(str):
    return str.lower().split()
