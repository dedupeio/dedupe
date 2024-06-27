#!/usr/bin/python
from __future__ import annotations

import abc
import re
import string
from itertools import product
from typing import TYPE_CHECKING

import dedupe.levenshtein as levenshtein
import dedupe.tfidf as tfidf
from dedupe.cpredicates import ngrams

# This allows to import predicate functions from this module
# and ensure backward compatibility.
from dedupe.predicate_functions import *  # noqa: F401, F403

if TYPE_CHECKING:
    from typing import AbstractSet, Any, Iterable, Literal, Mapping, Sequence

    from dedupe._typing import PredicateFunction, RecordDict
    from dedupe.index import Index


words = re.compile(r"[\w']+").findall

PUNCTABLE = str.maketrans("", "", string.punctuation)


def strip_punc(s: str) -> str:
    return s.translate(PUNCTABLE)


class NoIndexError(AttributeError):
    def __init__(self, *args) -> None:
        super().__init__(args[0])

        self.failing_record = None
        if len(args) > 1:
            self.failing_record = args[1]


class Predicate(abc.ABC):
    type: str
    __name__: str
    _cached_hash: int
    cover_count: int

    def __iter__(self):
        yield self

    def __repr__(self) -> str:
        return "{}: {}".format(self.type, self.__name__)

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
    def __call__(self, record: RecordDict, **kwargs) -> AbstractSet[str]:
        pass

    def __add__(self, other: Predicate) -> CompoundPredicate:
        if isinstance(other, CompoundPredicate):
            return CompoundPredicate((self,) + tuple(other))
        elif isinstance(other, Predicate):
            return CompoundPredicate((self, other))
        else:
            raise ValueError("Can only combine predicates")


class SimplePredicate(Predicate):
    type = "SimplePredicate"

    def __init__(self, func: PredicateFunction, field: str):
        self.func = func
        self.__name__ = "({}, {})".format(func.__name__, field)
        self.field = field

    def __call__(self, record: RecordDict, **kwargs) -> frozenset[str]:
        column = record[self.field]
        if column:
            return self.func(column)
        else:
            return frozenset()


class StringPredicate(SimplePredicate):
    def __call__(self, record: RecordDict, **kwargs) -> frozenset[str]:
        column: str = record[self.field]
        if column:
            return self.func(" ".join(strip_punc(column).split()))
        else:
            return frozenset()


class ExistsPredicate(Predicate):
    type = "ExistsPredicate"

    def __init__(self, field: str):
        self.__name__ = "(Exists, {})".format(field)
        self.field = field

    @staticmethod
    def func(column: Any) -> frozenset[Literal["0", "1"]]:
        if column:
            return frozenset(("1",))
        else:
            return frozenset(("0",))

    def __call__(self, record: RecordDict, **kwargs) -> frozenset[Literal["0", "1"]]:  # type: ignore
        column = record[self.field]
        return self.func(column)


class IndexPredicate(Predicate):
    field: str
    threshold: float
    index: Index | None
    _cache: dict[Any, frozenset[str]]

    def __init__(self, threshold: float, field: str):
        self.__name__ = "({}, {})".format(threshold, field)
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
    def reset(self) -> None: ...

    @abc.abstractmethod
    def initIndex(self) -> Index: ...

    def bust_cache(self) -> None:
        self._cache = {}

    @abc.abstractmethod
    def preprocess(self, doc: Any) -> Any: ...


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

    def __call__(self, record: RecordDict, **kwargs) -> frozenset[str]:
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
            return frozenset()
        else:
            return frozenset((str(block_key),))


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

    def __call__(
        self, record: RecordDict, target: bool = False, **kwargs
    ) -> frozenset[str]:
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
            result = frozenset(str(center) for center in centers)
            self._cache[(column, target)] = result
            return result
        else:
            return frozenset()


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

    def __call__(self, record: RecordDict, **kwargs) -> frozenset[str]:
        predicate_keys = [predicate(record, **kwargs) for predicate in self]
        return frozenset(
            ":".join(
                # must escape : to avoid confusion with : join separator
                b.replace(":", "\\:")
                for b in block_key
            )
            for block_key in product(*predicate_keys)
        )

    def __add__(self, other: Predicate) -> CompoundPredicate:  # type: ignore
        if isinstance(other, CompoundPredicate):
            return CompoundPredicate(tuple(self) + tuple(other))
        elif isinstance(other, Predicate):
            return CompoundPredicate(tuple(self) + (other,))
        else:
            raise ValueError("Can only combine predicates")
