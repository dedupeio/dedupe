#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import math
import itertools
import string
import abc

from doublemetaphone import doublemetaphone
from dedupe.cpredicates import ngrams, initials
import dedupe.tfidf as tfidf
import dedupe.levenshtein as levenshtein

from typing import Sequence, Callable, Any, Tuple, Set, Iterable
from dedupe._typing import RecordDict

words = re.compile(r"[\w']+").findall
integers = re.compile(r"\d+").findall
start_word = re.compile(r"^([\w']+)").match
start_integer = re.compile(r"^(\d+)").match
alpha_numeric = re.compile(r"(?=\w*\d)[a-zA-Z\d]+").findall

PUNCTABLE = str.maketrans("", "", string.punctuation)


def strip_punc(s):
    return s.translate(PUNCTABLE)


class Predicate(abc.ABC):
    def __iter__(self):
        yield self

    def __repr__(self):
        return "%s: %s" % (self.type, self.__name__)

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(repr(self))
            return h

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __len__(self):
        return 1

    @abc.abstractmethod
    def __call__(self, record, **kwargs):
        pass


class SimplePredicate(Predicate):
    type = "SimplePredicate"

    def __init__(self, func: Callable[[Any], Sequence[str]], field: str):
        self.func = func
        self.__name__ = "(%s, %s)" % (func.__name__, field)
        self.field = field

    def __call__(self, record: RecordDict, **kwargs) -> Iterable[str]:
        column = record[self.field]
        if column:
            return self.func(column)
        else:
            return ()

    def compounds_with(self, other):

        return True


class StringPredicate(SimplePredicate):
    def __call__(self, record: RecordDict, **kwargs):
        column = record[self.field]
        if column:
            return self.func(" ".join(strip_punc(column).split()))
        else:
            return ()

    def compounds_with(self, other):

        if other.field == self.field:
            return getattr(self.func,
                           'compounds_with_same_field',
                           True)

        return True


class ExistsPredicate(Predicate):
    type = "ExistsPredicate"

    def __init__(self, field):
        self.__name__ = "(Exists, %s)" % (field,)
        self.field = field
        self.compounds_with_same_field = False

    @staticmethod
    def func(column):
        if column:
            return ('1',)
        else:
            return ('0',)

    def __call__(self, record, **kwargs):
        column = record[self.field]
        return self.func(column)

    def compounds_with(self, other):

        if self.field == other.field:
            return False

        return True


class IndexPredicate(Predicate):
    def __init__(self, threshold, field):
        self.__name__ = '(%s, %s)' % (threshold, field)
        self.field = field
        self.threshold = threshold
        self.index = None
        self.compounds_with_same_field = False

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict['index'] = None
        return odict

    def __setstate__(self, d):
        self.__dict__.update(d)

        # backwards compatibility
        if not hasattr(self, 'index'):
            self.index = None

    def compounds_with(self, other):

        if other.field == self.field:
            if type(other) == type(self):
                return False

        return True

    def reset(self):
        ...

    def bust_cache(self):
        self._cache = {}


class CanopyPredicate(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canopy = {}
        self._cache = {}

    def freeze(self, records):
        self._cache = {record[self.field]: self(record) for record in records}
        self.canopy = {}
        self.index = None

    def reset(self):
        self._cache = {}
        self.canopy = {}
        self.index = None

    def __call__(self, record, **kwargs):
        block_key = None
        column = record[self.field]

        if column:
            if column in self._cache:
                return self._cache[column]

            doc = self.preprocess(column)

            try:
                doc_id = self.index._doc_to_id[doc]
            except AttributeError:
                raise AttributeError("Attempting to block with an index "
                                     "predicate without indexing records")

            if doc_id in self.canopy:
                block_key = self.canopy[doc_id]
            else:
                canopy_members = self.index.search(doc,
                                                   self.threshold)
                for member in canopy_members:
                    if member not in self.canopy:
                        self.canopy[member] = doc_id

                if canopy_members:
                    block_key = doc_id
                    self.canopy[doc_id] = doc_id
                else:
                    self.canopy[doc_id] = None

        if block_key is None:
            return []
        else:
            return [str(block_key)]


class SearchPredicate(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def freeze(self, records_1, records_2):
        self._cache = {(record[self.field], False): self(record, False)
                       for record in records_1}
        self._cache.update({(record[self.field], True): self(record, True)
                            for record in records_2})
        self.index = None

    def reset(self):
        self._cache = {}
        self.index = None

    def __call__(self, record, target=False, **kwargs):
        column = record[self.field]
        if column:
            if (column, target) in self._cache:
                return self._cache[(column, target)]
            else:
                doc = self.preprocess(column)

                try:
                    if target:
                        centers = [self.index._doc_to_id[doc]]
                    else:
                        centers = self.index.search(doc, self.threshold)
                except AttributeError:
                    raise AttributeError("Attempting to block with an index "
                                         "predicate without indexing records")
                result = [str(center) for center in centers]
                self._cache[(column, target)] = result
                return result
        else:
            return ()


class TfidfPredicate(IndexPredicate):
    def initIndex(self):
        self.reset()
        return tfidf.TfIdfIndex()


class TfidfCanopyPredicate(CanopyPredicate, TfidfPredicate):
    pass


class TfidfSearchPredicate(SearchPredicate, TfidfPredicate):
    pass


class TfidfTextPredicate(object):

    def preprocess(self, doc):
        return tuple(words(doc))


class TfidfSetPredicate(object):
    def preprocess(self, doc):
        return doc


class TfidfNGramPredicate(object):
    def preprocess(self, doc):
        return tuple(sorted(ngrams(" ".join(strip_punc(doc).split()), 2)))


class TfidfTextSearchPredicate(TfidfTextPredicate,
                               TfidfSearchPredicate):
    type = "TfidfTextSearchPredicate"


class TfidfSetSearchPredicate(TfidfSetPredicate,
                              TfidfSearchPredicate):
    type = "TfidfSetSearchPredicate"


class TfidfNGramSearchPredicate(TfidfNGramPredicate,
                                TfidfSearchPredicate):
    type = "TfidfNGramSearchPredicate"


class TfidfTextCanopyPredicate(TfidfTextPredicate,
                               TfidfCanopyPredicate):
    type = "TfidfTextCanopyPredicate"


class TfidfSetCanopyPredicate(TfidfSetPredicate,
                              TfidfCanopyPredicate):
    type = "TfidfSetCanopyPredicate"


class TfidfNGramCanopyPredicate(TfidfNGramPredicate,
                                TfidfCanopyPredicate):
    type = "TfidfNGramCanopyPredicate"


class LevenshteinPredicate(IndexPredicate):
    def initIndex(self):
        self.reset()
        return levenshtein.LevenshteinIndex()

    def preprocess(self, doc):
        return " ".join(strip_punc(doc).split())


class LevenshteinCanopyPredicate(CanopyPredicate, LevenshteinPredicate):
    type = "LevenshteinCanopyPredicate"


class LevenshteinSearchPredicate(SearchPredicate, LevenshteinPredicate):
    type = "LevenshteinSearchPredicate"


class CompoundPredicate(tuple):
    type = "CompoundPredicate"

    @property
    def __name__(self):
        return u'(%s)' % u', '.join(str(pred) for pred in self)

    def __call__(self, record, **kwargs):
        predicate_keys = [predicate(record, **kwargs)
                          for predicate in self]
        return [
            u':'.join(
                # must escape : to avoid confusion with : join separator
                b.replace(u':', u'\\:') for b in block_key
            )
            for block_key
            in itertools.product(*predicate_keys)
        ]


def wholeFieldPredicate(field: Any) -> Tuple[str]:
    """return the whole field"""
    return (str(field), )


wholeFieldPredicate.compounds_with_same_field = False  # type: ignore


def tokenFieldPredicate(field):
    """returns the tokens"""
    return set(words(field))


def firstTokenPredicate(field: str) -> Sequence[str]:
    first_token = start_word(field)
    if first_token:
        return first_token.groups()
    else:
        return ()


def commonIntegerPredicate(field: str) -> Set[str]:
    """return any integers"""
    return {str(int(i)) for i in integers(field)}


def alphaNumericPredicate(field: str) -> Set[str]:
    return set(alpha_numeric(field))


def nearIntegersPredicate(field: str) -> Set[str]:
    """return any integers N, N+1, and N-1"""
    ints = integers(field)
    near_ints = set()
    for char in ints:
        num = int(char)
        near_ints.add(str(num - 1))
        near_ints.add(str(num))
        near_ints.add(str(num + 1))

    return near_ints


def hundredIntegerPredicate(field: str) -> Set[str]:
    return {str(int(i))[:-2] + '00' for i in integers(field)}


def hundredIntegersOddPredicate(field: str) -> Set[str]:
    return {str(int(i))[:-2] + '0' + str(int(i) % 2) for i in integers(field)}


def firstIntegerPredicate(field: str) -> Sequence[str]:
    first_token = start_integer(field)
    if first_token:
        return first_token.groups()
    else:
        return ()


def ngramsTokens(field: Sequence[Any], n: int) -> Set[str]:
    grams = set()
    n_tokens = len(field)
    for i in range(n_tokens):
        for j in range(i + n, min(n_tokens, i + n) + 1):
            grams.add(' '.join(str(tok) for tok in field[i:j]))
    return grams


def commonTwoTokens(field: str) -> Set[str]:
    return ngramsTokens(field.split(), 2)


def commonThreeTokens(field: str) -> Set[str]:
    return ngramsTokens(field.split(), 3)


def fingerprint(field: str) -> Tuple[str]:
    return (u''.join(sorted(field.split())).strip(),)


def oneGramFingerprint(field: str) -> Tuple[str]:
    return (u''.join(sorted(set(ngrams(field.replace(' ', ''), 1)))).strip(),)


def twoGramFingerprint(field: str) -> Tuple[str, ...]:
    if len(field) > 1:
        return (u''.join(sorted(gram.strip() for gram
                                in set(ngrams(field.replace(' ', ''), 2)))),)
    else:
        return ()


def commonFourGram(field: str) -> Set[str]:
    """return 4-grams"""
    return set(ngrams(field.replace(' ', ''), 4))


def commonSixGram(field: str) -> Set[str]:
    """return 6-grams"""
    return set(ngrams(field.replace(' ', ''), 6))


def sameThreeCharStartPredicate(field: str) -> Tuple[str]:
    """return first three characters"""
    return initials(field.replace(' ', ''), 3)


def sameFiveCharStartPredicate(field: str) -> Tuple[str]:
    """return first five characters"""
    return initials(field.replace(' ', ''), 5)


def sameSevenCharStartPredicate(field: str) -> Tuple[str]:
    """return first seven characters"""
    return initials(field.replace(' ', ''), 7)


def suffixArray(field):
    n = len(field) - 4
    if n > 0:
        for i in range(0, n):
            yield field[i:]


suffixArray.compounds_with_same_field = False  # type: ignore


def sortedAcronym(field: str) -> Tuple[str]:
    return (''.join(sorted(each[0] for each in field.split())),)


def doubleMetaphone(field):
    return {metaphone for metaphone in doublemetaphone(field) if metaphone}


def metaphoneToken(field):
    return {metaphone_token for metaphone_token
            in itertools.chain(*(doublemetaphone(token)
                                 for token in set(field.split())))
            if metaphone_token}


def wholeSetPredicate(field_set):
    return (str(field_set),)


wholeSetPredicate.compounds_with_same_field = False  # type: ignore


def commonSetElementPredicate(field_set):
    """return set as individual elements"""
    return tuple([str(each) for each in field_set])


def commonTwoElementsPredicate(field):
    sequence = sorted(field)
    return ngramsTokens(sequence, 2)


def commonThreeElementsPredicate(field):
    sequence = sorted(field)
    return ngramsTokens(sequence, 3)


def lastSetElementPredicate(field_set):
    return (str(max(field_set)), )


def firstSetElementPredicate(field_set):
    return (str(min(field_set)), )


def magnitudeOfCardinality(field_set):
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
    if any(field):
        return (str([round(dim, digits) for dim in field]),)
    else:
        return ()


def orderOfMagnitude(field):
    if field > 0:
        return (str(int(round(math.log10(field)))), )
    else:
        return ()


def roundTo1(field):  # thanks http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    abs_num = abs(field)
    order = int(math.floor(math.log10(abs_num)))
    rounded = round(abs_num, -order)
    return (str(int(math.copysign(rounded, field))),)
