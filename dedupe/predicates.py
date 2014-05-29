#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import math

from dedupe.cpredicates import ngrams, initials

words = re.compile("[\w']+").findall
integers = re.compile("\d+").findall
start_word = re.compile("^[\w']+").findall
start_integer = re.compile("^\d+").findall


def wholeFieldPredicate(field):
    """return the whole field"""

    if field:
        return (field, )
    else:
        return ()

def tokenFieldPredicate(field):
    """returns the tokens"""
    return set(words(field))

def firstTokenPredicate(field) :
    first_token = start_word(field)
    return tuple(first_token)

def commonIntegerPredicate(field):
    """return any integers"""
    return set(integers(field))

def nearIntegersPredicate(field):
    """return any integers N, N+1, and N-1"""
    ints = integers(field)
    near_ints = set(ints)
    for char in ints :
        num = int(char)
        near_ints.add(unicode(num-1))
        near_ints.add(unicode(num+1))
        
    return near_ints

def firstIntegerPredicate(field) :
    first_token = start_integer(field)
    return tuple(first_token)
    
def commonFourGram(field):
    """return 4-grams"""
    return ngrams(field, 4)

def commonSixGram(field):
    """return 6-grams"""
    return ngrams(field, 6)


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
