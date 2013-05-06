#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import math

words = re.compile("[\w']+")
integers = re.compile("\d+")

def wholeFieldPredicate(field):
    """return the whole field"""

    if field:
        return (field, )
    else:
        return ()

def tokenFieldPredicate(field):
    """returns the tokens"""
    return tuple(words.findall(field))

def commonIntegerPredicate(field):
    """return any integers"""
    return tuple(integers.findall(field))

def nearIntegersPredicate(field):
    """return any integers N, N+1, and N-1"""
    ints = sorted([int(i) for i in integers.findall(field)])
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
    """return 6-grams"""
    return ngrams(field, 6)

def initials(field, n):
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
    """
    return (field[:n], ) if len(field) > n-1 else () 

def sameThreeCharStartPredicate(field):
    """return first three characters"""
    return initials(field, 3)

def sameFiveCharStartPredicate(field):
    """return first five characters"""
    return initials(field, 5)

def sameSevenCharStartPredicate(field):
    """return first seven characters"""
    return initials(field, 7)

def wholeSetPredicate(field_set):
    try:
        set_len = len(field_set)
    except TypeError:
        return tuple([field_set])
    
    if set_len == 0:
        return ()
    return(tuple(field_set))

def commonSetElementPredicate(field_set):
    """return set as individual elements"""
    try:
        set_len = len(field_set)
    except TypeError:
        return tuple([field_set])
    
    if set_len < 1:
        return ()

    return tuple(str(f) for f in field_set)

def roundToBase(x, base):
    """Given a float and a base, return the spanning values to the nearest base"""
    base_log = int(-1 * math.log10(base))
    base_ceil = round(float(base * math.ceil(float(x) / base)), base_log)
    base_floor = round(float(base * math.floor(float(x) / base)), base_log)
    return [base_floor, base_ceil]

def checkEqual(iter):
    if len(set(iter)) > 1:
        return False
    else:
        return True

def expandGrid(coord_tuple, base):
    """
    If the coordinate pair defining a grid is equal, expand it out so
    that the grid is 2 * base wide.
    """
    if len(set(coord_tuple)) == 1:
        coord_tuple[0] -= base
        coord_tuple[1] += base
    return coord_tuple        
    
def latLongGridPredicate(field, base=0.1):
    """
    Given a lat / long pair, return the grid coordinates
    at the nearest base value.
    e.g., (42.3, -5.4) returns a grid at 0.1 degree resolution of
    0.1 degrees of latitude ~ 7km, so this is effectively a 14km lat grid.
    This is imprecise for longitude, since 1 degree of longitude is 0km at the poles,
    and up to 111km at the equator. But it should be reasonably precise given some
    prior logical block (e.g., country).
    """
    if field[0] == 0 and field[1] == 0:
        return ()
    lat_grid = roundToBase(field[0], base)
    long_grid = roundToBase(field[1], base)

    if checkEqual(lat_grid):
        lat_grid = expandGrid(lat_grid, base)
    if checkEqual(long_grid):
        long_grid = expandGrid(long_grid, base)
    
    grid = (tuple(lat_grid), tuple(long_grid))
    return str(grid)
