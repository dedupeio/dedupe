#!python
#cython: boundscheck=False, wraparound=False, profile=True

from libc.math cimport sqrt, log10, pow
# import numpy as np
cimport numpy as np
import pdb

def calculateDocumentFrequency(iterable_list, idf_threshold):
    """
    Given a list of either values or iterables (set/list/tuple),
    return the idf of each unique value in the form of a value:idf
    dict
    
    This assumes that the iterables are unique sets, so the
    frequency of any given term in a document is not > 1.

    Terms with frequencies greater than n_documents * idf_threshold are discarded.
    """
    cdef float n_docs = len(iterable_list)
    cdef tf_dict = {}
    for i in iterable_list:
        ## Pick up non-string iterables only
        if hasattr(i, '__iter__'):
            for j in i:
                if j in tf_dict:
                    tf_dict[j] += 1.0
                else:
                    tf_dict[j] = 1.0
        else:
            if i in tf_dict:
                tf_dict[i] += 1.0
            else:
                tf_dict[i] = 1.0

    cdef idf_dict = {}

    cdef float idf
    
    for t, v in tf_dict.iteritems():
        if v > n_docs * idf_threshold :
            print t , v
            idf = 0
        else :
            idf = log10(n_docs / v)
        idf_dict[t] = idf
                       
    return idf_dict

def sum_dict_subset(frozenset s, dict d):
    cdef float out = 0
    cdef float val
    for k in s:
        val = pow(d[k], 2)
        out += val
    return out

def sum_dict_set_intersect(frozenset s1, frozenset s2, dict d):
    cdef float out = 0
    cdef float val
    for k in s1.intersection(s2):
        val = pow(d[k], 2)
        out += val
    return out

def createCosineSimilarity(iterable_list, idf_threshold=0.05):
    """
    Closure to generate a cosine similarity function with tfidf weights.
    Note that this assumes sets, such that any given value occurs in s1 or s2 at
    most once (i.e., tfidf == tf b/c tf == 1).

    Terms with frequencies greater than n_documents * idf_threshold are discarded.
    """
    document_tf = calculateDocumentFrequency(iterable_list, idf_threshold)
    dfd = document_tf

    def cosine(s1, s2):
        cdef float numer = sum_dict_set_intersect(s1, s2, dfd)
        cdef float denom_a = sqrt(sum_dict_subset(s1, dfd))
        cdef float denom_b = sqrt(sum_dict_subset(s2, dfd))
        cdef float denom = denom_a * denom_b

        if denom == 0:
            return 0.0
        else:
            return numer / denom

    return cosine

class CosineSimilarity :
    """
    Defines a class version of the closure. The pure closure
    version is slightly faster but can't be saved (pickled) in settings file.

    Terms with frequencies greater than n_documents * idf_threshold are discarded.
    """

    def __init__(self, iterable_list, idf_threshold=0.05):
        self.iterable_list = iterable_list
        self.idf_threshold = idf_threshold
        self.dfd = calculateDocumentFrequency(self.iterable_list,
                                              self.idf_threshold
                                              )

    def __call__(self, s1, s2):
        cdef float numer = sum_dict_set_intersect(s1, s2, self.dfd)
        cdef float denom_a = sqrt(sum_dict_subset(s1, self.dfd))
        cdef float denom_b = sqrt(sum_dict_subset(s2, self.dfd))
        cdef float denom = denom_a * denom_b

        if denom == 0:
            return 0.0
        else:
            return numer / denom



