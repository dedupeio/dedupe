#!python
#cython: boundscheck=False, wraparound=False, profile=True

from libc.math cimport sqrt, log10, pow
# import numpy as np
cimport numpy as np

def calculateDocumentFrequency(iterable_list):
    """
    Given a list of either values or iterables (set/list/tuple),
    return the idf of each unique value in the form of a value:idf
    dict
    
    This assumes that the iterables are unique sets, so the
    frequency of any given term in a document is not > 1.
    """
    cdef float n_docs = len(iterable_list)
    cdef tf_dict = {}
    for i in iterable_list:
        ## Pick up non-string iterables only
        if hasattr(i, '__iter__'):
            for j in i:
                if j in tf_dict:
                    tf_dict[j] += 1
                else:
                    tf_dict[j] = 1
        else:
            if i in tf_dict:
                tf_dict[i] += 1
            else:
                tf_dict[i] = 1

    cdef idf_dict = {}
    cdef float idf
    for t, v in tf_dict.iteritems():
        idf = log10(n_docs / v)
        idf_dict[t] = idf
                       
    return idf_dict

def sum_dict_subset(s, dict d):
    cdef float out = 0
    cdef float val
    for k in s:
        val = pow(d[k], 2)
        out += val
    return out

def sum_dict_set_intersect(s1, s2, dict d):
    cdef float out = 0
    cdef float val
    for k in s1:
        if k in s2:
            val = pow(d[k], 2)
            out += val
    return out

def createCosineSimilarity(iterable_list):
    """
    Closure to generate a cosine similarity function with tfidf weights.
    Note that this assumes sets, such that any given value occurs in s1 or s2 at
    most once (i.e., tfidf == tf b/c tf == 1).
    """
    document_tf = calculateDocumentFrequency(iterable_list)
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

class CosineSimilarity:
    """
    Defines a class version of the closure. The pure closure
    version is slightly faster but can't be saved (pickled) in settings file.
    """
    def __init__(self, iterable_list):
        self.iterable_list = iterable_list
        self.dfd = calculateDocumentFrequency(self.iterable_list)

    def __call__(self, s1, s2):
        cdef float numer = sum_dict_set_intersect(s1, s2, self.dfd)
        cdef float denom_a = sqrt(sum_dict_subset(s1, self.dfd))
        cdef float denom_b = sqrt(sum_dict_subset(s2, self.dfd))
        cdef float denom = denom_a * denom_b
        
        if denom == 0:
            return 0.0
        else:
            return numer / denom
