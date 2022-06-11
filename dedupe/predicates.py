#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
import itertools
import math
import re
import string
from typing import TYPE_CHECKING

from doublemetaphone import doublemetaphone

import dedupe.levenshtein as levenshtein
import dedupe.tfidf as tfidf
from dedupe.cpredicates import initials, ngrams

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Mapping, Sequence

    from dedupe._typing import Literal, RecordDict
    from dedupe.index import Index


words = re.compile(r"[\w']+").findall
integers = re.compile(r"\d+").findall
start_word = re.compile(r"^([\w']+)").match
two_start_words = re.compile(r"^([\w']+\s+[\w']+)").match
start_integer = re.compile(r"^(\d+)").match
alpha_numeric = re.compile(r"(?=[a-zA-Z]*\d)[a-zA-Z\d]+").findall

PUNCTABLE = str.maketrans("", "", string.punctuation)


class NoIndexError(AttributeError):
    def __init__(self, *args) -> None:
        super().__init__(args[0])

        self.failing_record = None
        if len(args) > 1:
            self.failing_record = args[1]


def strip_punc(s: str) -> str:
    return s.translate(PUNCTABLE)


class Predicate(abc.ABC):
    type: str
    __name__: str
    _cached_hash: int
    cover_count: int

    def __iter__(self):
        yield self

    def __repr__(self) -> str:
        return "%s: %s" % (self.type, self.__name__)

    def __hash__(self) -> int:
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(repr(self))
            return h

    def __eq__(self, other: Any) -> bool:
        return repr(self) == repr(other)

    def __len__(self) -> int:
        return 1

    @abc.abstractmethod
    def __call__(self, record: RecordDict, **kwargs) -> Iterable[str]:
        pass

    def __add__(self, other: "Predicate") -> "CompoundPredicate":

        if isinstance(other, CompoundPredicate):
            return CompoundPredicate((self,) + tuple(other))
        elif isinstance(other, Predicate):
            return CompoundPredicate((self, other))
        else:
            raise ValueError("Can only combine predicates")


class SimplePredicate(Predicate):
    type = "SimplePredicate"

    def __init__(self, func: Callable[[Any], Iterable[str]], field: str):
        self.func = func
        self.__name__ = "(%s, %s)" % (func.__name__, field)
        self.field = field

    def __call__(self, record: RecordDict, **kwargs) -> Iterable[str]:
        column = record[self.field]
        if column:
            return self.func(column)
        else:
            return ()


class StringPredicate(SimplePredicate):
    def __call__(self, record: RecordDict, **kwargs) -> Iterable[str]:
        column: str = record[self.field]
        if column:
            return self.func(" ".join(strip_punc(column).split()))
        else:
            return ()


class ExistsPredicate(Predicate):
    type = "ExistsPredicate"

    def __init__(self, field: str):
        self.__name__ = "(Exists, %s)" % (field,)
        self.field = field

    @staticmethod
    def func(column: Any) -> tuple[Literal["0", "1"]]:
        if column:
            return ("1",)
        else:
            return ("0",)

    def __call__(self, record: RecordDict, **kwargs) -> tuple[Literal["0", "1"]]:
        column = record[self.field]
        return self.func(column)


class IndexPredicate(Predicate):
    field: str
    threshold: float
    index: Index | None
    _cache: dict[Any, list[str]]

    def __init__(self, threshold: float, field: str):
        self.__name__ = "(%s, %s)" % (threshold, field)
        self.field = field
        self.threshold = threshold
        self.index = None

    def __getstate__(self) -> dict[str, Any]:
        odict = self.__dict__.copy()
        odict["index"] = None
        odict["_cache"] = {}
        if "canopy" in odict:
            odict["canopy"] = {}
        return odict

    def __setstate__(self, d: Mapping[str, Any]) -> None:
        self.__dict__.update(d)

        # backwards compatibility
        if not hasattr(self, "index"):
            self.index = None

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @abc.abstractmethod
    def initIndex(self) -> Index:
        ...

    def bust_cache(self) -> None:
        self._cache = {}

    @abc.abstractmethod
    def preprocess(self, doc: Any) -> Any:
        ...


class CanopyPredicate(IndexPredicate):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.canopy: dict[Any, int | None] = {}
        self._cache = {}

    def freeze(self, records: Iterable[RecordDict]) -> None:
        self._cache = {record[self.field]: self(record) for record in records}
        self.canopy = {}
        self.index = None

    def reset(self) -> None:
        self._cache = {}
        self.canopy = {}
        self.index = None

    def __call__(self, record: RecordDict, **kwargs) -> list[str]:

        block_key = None
        column = record[self.field]

        if column:
            if column in self._cache:
                return self._cache[column]

            # we need to check for the index here, instead of the very
            # beginning because freezing predicates removes the index
            try:
                assert self.index is not None
            except AssertionError:
                raise NoIndexError(
                    "Attempting to block with an index "
                    "predicate without indexing records",
                    record,
                )

            doc = self.preprocess(column)

            doc_id = self.index._doc_to_id[doc]

            if doc_id in self.canopy:
                block_key = self.canopy[doc_id]
            else:
                canopy_members = self.index.search(doc, self.threshold)
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


class SearchPredicate(IndexPredicate):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache = {}

    def freeze(
        self, records_1: Iterable[RecordDict], records_2: Iterable[RecordDict]
    ) -> None:
        self._cache = {
            (record[self.field], False): self(record, False) for record in records_1
        }
        self._cache.update(
            {(record[self.field], True): self(record, True) for record in records_2}
        )
        self.index = None

    def reset(self) -> None:
        self._cache = {}
        self.index = None

    def __call__(self, record: RecordDict, target: bool = False, **kwargs) -> list[str]:

        column = record[self.field]
        if column:
            if (column, target) in self._cache:
                return self._cache[(column, target)]

            # we need to check for the index here, instead of the very
            # beginning because freezing predicates removes the index
            try:
                assert self.index is not None
            except AssertionError:
                raise NoIndexError(
                    "Attempting to block with an index "
                    "predicate without indexing records",
                    record,
                )

            doc = self.preprocess(column)

            if target:
                centers = [self.index._doc_to_id[doc]]
            else:
                centers = self.index.search(doc, self.threshold)
            result = [str(center) for center in centers]
            self._cache[(column, target)] = result
            return result
        else:
            return []


class TfidfPredicate(IndexPredicate):
    def initIndex(self) -> Index:
        self.reset()
        return tfidf.TfIdfIndex()


class TfidfCanopyPredicate(CanopyPredicate, TfidfPredicate):
    pass


class TfidfSearchPredicate(SearchPredicate, TfidfPredicate):
    pass


class TfidfTextPredicate(IndexPredicate):
    def preprocess(self, doc: str) -> Sequence[str]:
        return tuple(words(doc))


class TfidfSetPredicate(IndexPredicate):
    def preprocess(self, doc: Any) -> Any:
        return doc


class TfidfNGramPredicate(IndexPredicate):
    def preprocess(self, doc: str) -> Sequence[str]:
        return tuple(sorted(ngrams(" ".join(strip_punc(doc).split()), 2)))


class TfidfTextSearchPredicate(TfidfTextPredicate, TfidfSearchPredicate):
    type = "TfidfTextSearchPredicate"


class TfidfSetSearchPredicate(TfidfSetPredicate, TfidfSearchPredicate):
    type = "TfidfSetSearchPredicate"


class TfidfNGramSearchPredicate(TfidfNGramPredicate, TfidfSearchPredicate):
    type = "TfidfNGramSearchPredicate"


class TfidfTextCanopyPredicate(TfidfTextPredicate, TfidfCanopyPredicate):
    type = "TfidfTextCanopyPredicate"


class TfidfSetCanopyPredicate(TfidfSetPredicate, TfidfCanopyPredicate):
    type = "TfidfSetCanopyPredicate"


class TfidfNGramCanopyPredicate(TfidfNGramPredicate, TfidfCanopyPredicate):
    type = "TfidfNGramCanopyPredicate"


class LevenshteinPredicate(IndexPredicate):
    def initIndex(self) -> Index:
        self.reset()
        return levenshtein.LevenshteinIndex()

    def preprocess(self, doc: str) -> str:
        return " ".join(strip_punc(doc).split())


class LevenshteinCanopyPredicate(CanopyPredicate, LevenshteinPredicate):
    type = "LevenshteinCanopyPredicate"


class LevenshteinSearchPredicate(SearchPredicate, LevenshteinPredicate):
    type = "LevenshteinSearchPredicate"


class CompoundPredicate(tuple, Predicate):
    type = "CompoundPredicate"

    def __hash__(self) -> int:
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(frozenset(self))
            return h

    def __eq__(self, other: Any) -> bool:
        return frozenset(self) == frozenset(other)

    def __call__(self, record: RecordDict, **kwargs) -> list[str]:
        predicate_keys = [predicate(record, **kwargs) for predicate in self]
        return [
            ":".join(
                # must escape : to avoid confusion with : join separator
                b.replace(":", "\\:")
                for b in block_key
            )
            for block_key in itertools.product(*predicate_keys)
        ]

    def __add__(self, other: Predicate) -> "CompoundPredicate":  # type: ignore

        if isinstance(other, CompoundPredicate):
            return CompoundPredicate(tuple(self) + tuple(other))
        elif isinstance(other, Predicate):
            return CompoundPredicate(tuple(self) + (other,))
        else:
            raise ValueError("Can only combine predicates")


def wholeFieldPredicate(field: Any) -> tuple[str]:
    """return the whole field"""
    return (str(field),)


def tokenFieldPredicate(field: str) -> set[str]:
    """returns the tokens"""
    return set(words(field))


def firstTokenPredicate(field: str) -> Sequence[str]:
    first_token = start_word(field)
    if first_token:
        return first_token.groups()
    else:
        return ()


def firstTwoTokensPredicate(field: str) -> Sequence[str]:
    first_two_tokens = two_start_words(field)
    if first_two_tokens:
        return first_two_tokens.groups()
    else:
        return ()


def commonIntegerPredicate(field: str) -> set[str]:
    """return any integers"""
    return {str(int(i)) for i in integers(field)}


def alphaNumericPredicate(field: str) -> set[str]:
    return set(alpha_numeric(field))


def nearIntegersPredicate(field: str) -> set[str]:
    """return any integers N, N+1, and N-1"""
    ints = integers(field)
    near_ints = set()
    for char in ints:
        num = int(char)
        near_ints.add(str(num - 1))
        near_ints.add(str(num))
        near_ints.add(str(num + 1))

    return near_ints


def hundredIntegerPredicate(field: str) -> set[str]:
    return {str(int(i))[:-2] + "00" for i in integers(field)}


def hundredIntegersOddPredicate(field: str) -> set[str]:
    return {str(int(i))[:-2] + "0" + str(int(i) % 2) for i in integers(field)}


def firstIntegerPredicate(field: str) -> Sequence[str]:
    first_token = start_integer(field)
    if first_token:
        return first_token.groups()
    else:
        return ()


def ngramsTokens(field: Sequence[Any], n: int) -> set[str]:
    grams = set()
    n_tokens = len(field)
    for i in range(n_tokens):
        for j in range(i + n, min(n_tokens, i + n) + 1):
            grams.add(" ".join(str(tok) for tok in field[i:j]))
    return grams


def commonTwoTokens(field: str) -> set[str]:
    return ngramsTokens(field.split(), 2)


def commonThreeTokens(field: str) -> set[str]:
    return ngramsTokens(field.split(), 3)


def fingerprint(field: str) -> tuple[str]:
    return ("".join(sorted(field.split())).strip(),)


def oneGramFingerprint(field: str) -> tuple[str]:
    return ("".join(sorted(set(ngrams(field.replace(" ", ""), 1)))).strip(),)


def twoGramFingerprint(field: str) -> tuple[str, ...]:
    if len(field) > 1:
        return (
            "".join(
                sorted(gram.strip() for gram in set(ngrams(field.replace(" ", ""), 2)))
            ),
        )
    else:
        return ()


def commonFourGram(field: str) -> set[str]:
    """return 4-grams"""
    return set(ngrams(field.replace(" ", ""), 4))


def commonSixGram(field: str) -> set[str]:
    """return 6-grams"""
    return set(ngrams(field.replace(" ", ""), 6))


def sameThreeCharStartPredicate(field: str) -> tuple[str]:
    """return first three characters"""
    return initials(field.replace(" ", ""), 3)


def sameFiveCharStartPredicate(field: str) -> tuple[str]:
    """return first five characters"""
    return initials(field.replace(" ", ""), 5)


def sameSevenCharStartPredicate(field: str) -> tuple[str]:
    """return first seven characters"""
    return initials(field.replace(" ", ""), 7)


def suffixArray(field: str) -> Iterable[str]:
    n = len(field) - 4
    if n > 0:
        for i in range(0, n):
            yield field[i:]


def sortedAcronym(field: str) -> tuple[str]:
    return ("".join(sorted(each[0] for each in field.split())),)


def doubleMetaphone(field: str) -> set[str]:
    return {metaphone for metaphone in doublemetaphone(field) if metaphone}


def metaphoneToken(field: str) -> set[str]:
    return {
        metaphone_token
        for metaphone_token in itertools.chain(
            *(doublemetaphone(token) for token in set(field.split()))
        )
        if metaphone_token
    }


def wholeSetPredicate(field_set: Any) -> tuple[str]:
    return (str(field_set),)


def commonSetElementPredicate(field_set: str) -> tuple[str, ...]:
    """return set as individual elements"""
    return tuple([str(each) for each in field_set])


def commonTwoElementsPredicate(field: str) -> set[str]:
    sequence = sorted(field)
    return ngramsTokens(sequence, 2)


def commonThreeElementsPredicate(field: str) -> set[str]:
    sequence = sorted(field)
    return ngramsTokens(sequence, 3)


def lastSetElementPredicate(field_set: Sequence[Any]) -> tuple[str]:
    return (str(max(field_set)),)


def firstSetElementPredicate(field_set: Sequence[Any]) -> tuple[str]:
    return (str(min(field_set)),)


def magnitudeOfCardinality(field_set: Sequence[Any]) -> tuple[str, ...]:
    return orderOfMagnitude(len(field_set))


def latLongGridPredicate(field: tuple[float], digits: int = 1) -> tuple[str, ...]:
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


def orderOfMagnitude(field: int | float) -> tuple[str, ...]:
    if field > 0:
        return (str(int(round(math.log10(field)))),)
    else:
        return ()


def roundTo1(
    field: float,
) -> tuple[
    str
]:  # thanks http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    abs_num = abs(field)
    order = int(math.floor(math.log10(abs_num)))
    rounded = round(abs_num, -order)
    return (str(int(math.copysign(rounded, field))),)
