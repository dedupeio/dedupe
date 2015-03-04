#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import math
import itertools

from metaphone import doublemetaphone
from dedupe.cpredicates import ngrams, initials
import dedupe.tfidf as tfidf

words = re.compile(r"[\w']+").findall
integers = re.compile(r"\d+").findall
start_word = re.compile(r"^([\w']+)").match
start_integer = re.compile(r"^(\d+)").match

class Predicate(object) :
    def __iter__(self) :
        yield self
        
    def __repr__(self) :
        return "%s: %s" % (self.type, self.__name__)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other) :
        return repr(self) == repr(other)

class SimplePredicate(Predicate) :
    type = "SimplePredicate"

    def __init__(self, func, field) :
        self.func = func
        self.__name__ = "(%s, %s)" % (func.__name__, field)
        self.field = field

    def __call__(self, record) :
        column = record[self.field]
        if column :
            return self.func(column)
        else :
            return ()

class ExistsPredicate(Predicate) :
    type = "ExistsPredicate"

    def __init__(self, field) :
        self.__name__ = "(Exists, %s)" % (field,)
        self.field = field


    @staticmethod
    def func(column) :
        if column :
            return ('1',)
        else :
            return ('0',)


    def __call__(self, record) :
        column = record[self.field]
        return self.func(column)


class IndexPredicate(Predicate) :
    def __init__(self, threshold, field):
        self.__name__ = '(%s, %s)' % (threshold, field)
        self.field = field
        self.threshold = threshold
        self.index = None

    def __getstate__(self):

        result = self.__dict__.copy()

        return {'__name__': result['__name__'],
                'field' : result['field'],
                'threshold' : result['threshold']}

    def __setstate__(self, d) :
        self.__dict__ = d
        self.index = None

class TfidfPredicate(IndexPredicate):
    def initIndex(self, stop_words) :
        return tfidf.TfIdfIndex(stop_words)

    def __call__(self, record) :
        column = record[self.field]
        if column :
            try :
                centers = self.index.search(self.preprocess(column), 
                                            self.threshold)

                l_unicode = unicode
                return [l_unicode(center) for center in centers]
            except :
                raise AttributeError("Attempting to block with an index "
                                     "predicate without indexing records")

        else :
            return ()

class TfidfTextPredicate(TfidfPredicate) :
    type = "TfidfTextPredicate"

    rx = re.compile(r"(?u)\w+[\w*?]*")

    def preprocess(self, doc) :
        return tuple(self.rx.findall(doc))

class TfidfSetPredicate(TfidfPredicate) :
    type = "TfidfSetPredicate"

    def preprocess(self, doc) :
        return doc

class TfidfNGramPredicate(TfidfPredicate) :
    type = "TfidfNGramPredicate"

    def preprocess(self, doc) :
        return tuple(ngrams(doc.replace(' ', ''), 2))

class CompoundPredicate(Predicate) :
    type = "CompoundPredicate"

    def __init__(self, predicates) :
        self.predicates = predicates
        self.__name__ = u'(%s)' % u', '.join([unicode(pred)
                                              for pred in 
                                              predicates])

    def __iter__(self) :
        for pred in self.predicates :
            yield pred

    def __call__(self, record) :
        predicate_keys = [predicate(record)
                          for predicate in self.predicates]
        return [u':'.join(block_key)
                for block_key
                in itertools.product(*predicate_keys)]
 

def wholeFieldPredicate(field):
    """return the whole field"""
    return (unicode(field), )

def tokenFieldPredicate(field):
    """returns the tokens"""
    return set(words(field))

def firstTokenPredicate(field) :
    first_token = start_word(field)
    if first_token :
        return first_token.groups()
    else :
        return ()

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
    if first_token :
        return first_token.groups()
    else :
        return ()


def ngramsTokens(field, n) :
    grams = set([])
    n_tokens = len(field)
    for i in range(n_tokens):
        for j in range(i+n, min(n_tokens, i+n)+1):
            grams.add(' '.join(unicode(tok) for tok in field[i:j]))
    return grams


def commonTwoTokens(field) :
    return ngramsTokens(field.split(), 2)

def commonThreeTokens(field) :
    return ngramsTokens(field.split(), 3)

def fingerprint(field) :
    return (u''.join(sorted(field.split())).strip(),)

def oneGramFingerprint(field) :
    return (u''.join(sorted(set(ngrams(field.replace(' ', ''), 1)))).strip(),)

def twoGramFingerprint(field) :
    if len(field) > 1 :
        return (u''.join(sorted(gram.strip() for gram 
                                in set(ngrams(field.replace(' ', ''), 2)))),)
    else :
        return ()
    
def commonFourGram(field):
    """return 4-grams"""
    return set(ngrams(field.replace(' ', ''), 4))

def commonSixGram(field):
    """return 6-grams"""
    return set(ngrams(field.replace(' ', ''), 6))

def sameThreeCharStartPredicate(field):
    """return first three characters"""
    return initials(field.replace(' ', ''), 3)

def sameFiveCharStartPredicate(field):
    """return first five characters"""
    return initials(field.replace(' ', ''), 5)

def sameSevenCharStartPredicate(field):
    """return first seven characters"""
    return initials(field.replace(' ',''), 7)

def suffixArray(field) :
    field = field.replace(' ', '')
    n = len(field) - 4
    if n > 0 :
        for i in range(0, n) :
            yield field[i:]

def sortedAcronym(field) :
    return (''.join(sorted(each[0] for each in field.split())),)

def doubleMetaphone(field) :
    return [metaphone for metaphone in doublemetaphone(field) if metaphone]

def metaphoneToken(field) :
    return set([metaphone_token for metaphone_token 
                in itertools.chain(*(doublemetaphone(token) 
                                     for token in set(field.split())))
                if metaphone_token])

def existsPredicate(field) :
    try :
        if any(field) :
            return (u'1',)
        else :
            return (u'0',)
    except TypeError :
        if field :
            return (u'1',)
        else :
            return (u'0',)

def wholeSetPredicate(field_set):
    return (unicode(field_set),)

def commonSetElementPredicate(field_set):
    """return set as individual elements"""
    return tuple([unicode(each) for each in field_set])

def commonTwoElementsPredicate(field) :
    l = sorted(field)
    return ngramsTokens(l, 2)

def commonThreeElementsPredicate(field) :
    l = sorted(field)
    return ngramsTokens(l, 3)

def lastSetElementPredicate(field_set) :
    return (unicode(max(field_set)), )

def firstSetElementPredicate(field_set) :
    return (unicode(min(field_set)), )

def magnitudeOfCardinality(field_set) :
    return orderOfMagnitude(len(field_set))

def latLongGridPredicate(field, digits=1):
    """
    Given a lat / long pair, return the grid coordinates at the
    nearest base value.  e.g., (42.3, -5.4) returns a grid at 0.1
    degree resolution of 0.1 degrees of latitude ~ 7km, so this is
    effectively a 14km lat grid.  This is imprecise for longitude,
    since 1 degree of longitude is 0km at the poles, and up to 111km
    at the equator. But it should be reasonably precise given some
    prior logical block (e.g., country).
    """
    if any(field) :
        return (unicode([round(dim, digits) for dim in field]),)
    else :
        return ()

def orderOfMagnitude(field) :
    if field > 0 :
        return (unicode(int(round(math.log10(field)))), )
    else :
        return ()

def roundTo1(field) : # thanks http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    abs_num = abs(field)
    order = int(math.floor(math.log10(abs_num)))
    rounded = round(abs_num, -order)
    return (unicode(int(math.copysign(rounded, field))),)
        
