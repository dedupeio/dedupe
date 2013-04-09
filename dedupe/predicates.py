#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

def tokenFieldPredicate(field):
    """returns the tokens"""
    return tuple(field.split())

def commonIntegerPredicate(field):
    """"return any integers"""
    return tuple(re.findall("\d+", field))

def nearIntegersPredicate(field):
    """return any integers N, N+1, and N-1"""
    ints = sorted([int(i) for i in re.findall("\d+", field)])
    near_ints = set([])
    [near_ints.update((i - 1, i, i + 1)) for i in ints]
    return tuple(near_ints)

def ngrams(field, n):
    """ngrams returns all unique, contiguous sequences of n characters
    of a given field.
        
    :param field: the string to be 
    :param n: the number of characters to be included in each gram
    
    usage:
    >>> from dedupe.dedupe.predicated import ngrams
    >>> ngrams("deduplicate", 3)
    ('ded', 'edu', 'dup', 'upl', 'pli', 'lic', 'ica', 'cat', 'ate')
    """
    return tuple([field[pos:pos + n] for pos in xrange(len(field) - n + 1)])
    
def commonFourGram(field):
    """return 4-grams"""
    return ngrams(field, 4)

def commonSixGram(field):
    """"return 6-grams"""
    return ngrams(field, 6)

def initials(field, n=None):
    """predicate which returns first a tuple containing
    the first n chars of a field if and only if the
    field contains at least n characters, or an empty
    tuple otherwise.
    
    :param field: the string 
    :type n: int, default None
    
    usage:
    >>> initials("dedupe", 7)
    ()
    >>> initials("deduplication", 7)
    ('dedupli', )
    >>> initials("noslice")
    ('noslice', )
    """
    return (field[:n], ) if not n or len(field) > n-1 else () 

def wholeFieldPredicate(field):
    """return the whole field
    consider replacing with initials(field)
    """
    return (field, ) if field else ()

def sameThreeCharStartPredicate(field):
    """return first three characters"""
    return initials(field, 3)

def sameFiveCharStartPredicate(field):
    """return first five characters"""
    return initials(field, 5)

def sameSevenCharStartPredicate(field):
    """return first seven characters"""
    return initials(field, 7)
