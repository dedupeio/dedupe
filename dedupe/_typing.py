import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy
import numpy.typing

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, TypedDict
else:
    from typing_extensions import Literal, Protocol, TypedDict


if TYPE_CHECKING:
    from dedupe.predicates import Predicate


RecordDict = Mapping[str, Any]
RecordID = Union[int, str]
RecordIDDType = Union[Type[int], Tuple[Type[str], Literal[256]]]
RecordIDPair = Union[Tuple[int, int], Tuple[str, str]]
RecordInt = Tuple[int, RecordDict]
RecordStr = Tuple[str, RecordDict]
Record = Union[RecordInt, RecordStr]
RecordPairInt = Tuple[RecordInt, RecordInt]
RecordPairStr = Tuple[RecordStr, RecordStr]
RecordPairs = Union[Iterator[RecordPairInt], Iterator[RecordPairStr]]
BlockInt = List[RecordPairInt]
BlockStr = List[RecordPairStr]
Block = Union[RecordPairInt, RecordPairStr]
BlocksInt = Iterator[BlockInt]
BlocksStr = Iterator[BlockStr]
Blocks = Union[BlocksInt, BlocksStr]
ClusterInt = Tuple[
    Tuple[int, ...], Union[numpy.typing.NDArray[numpy.float_], Tuple[float, ...]]
]
ClusterStr = Tuple[
    Tuple[str, ...], Union[numpy.typing.NDArray[numpy.float_], Tuple[float, ...]]
]
ClustersInt = Iterable[ClusterInt]
ClustersStr = Iterable[ClusterStr]
Clusters = Union[ClustersInt, ClustersStr]

DataInt = Mapping[int, RecordDict]
DataStr = Mapping[str, RecordDict]
Data = Union[DataInt, DataStr]

RecordDictPair = Tuple[RecordDict, RecordDict]
RecordDictPairs = List[RecordDictPair]
ArrayLinks = Iterable[numpy.ndarray]
TupleLinksInt = Iterable[Tuple[Tuple[int, int], float]]
TupleLinksStr = Iterable[Tuple[Tuple[str, str], float]]
TupleLinks = Union[TupleLinksInt, TupleLinksStr]
Links = Union[ArrayLinks, TupleLinks]
LookupResultsInt = Iterable[Tuple[int, Tuple[Tuple[int, float], ...]]]
LookupResultsStr = Iterable[Tuple[str, Tuple[Tuple[str, float], ...]]]
LookupResults = Union[LookupResultsInt, LookupResultsStr]
JoinConstraint = Literal["one-to-one", "many-to-one", "many-to-many"]
Comparator = Callable[[Any, Any], Union[Union[int, float], Sequence[Union[int, float]]]]
Scores = Union[numpy.memmap, numpy.ndarray]
Labels = List[Literal[0, 1]]
LabelsLike = Iterable[Literal[0, 1]]
Cover = Dict["Predicate", FrozenSet[int]]
ComparisonCoverInt = Dict["Predicate", FrozenSet[Tuple[int, int]]]
ComparisonCoverStr = Dict["Predicate", FrozenSet[Tuple[str, str]]]
ComparisonCover = Union[ComparisonCoverInt, ComparisonCoverStr]
PredicateFunction = Callable[[Any], FrozenSet[str]]

VariableDefinition = TypedDict(
    "VariableDefinition",
    {
        "type": str,
        "field": str,
        "variable name": str,
        "corpus": Iterable[Union[str, Collection[str]]],
        "comparator": Callable[
            [Any, Any], Union[int, float]
        ],  # a custom comparator can only return a single float or int, not a sequence of numbers
        "categories": List[str],
        "interaction variables": List[str],
        "has missing": bool,
        "name": str,
    },
    total=False,
)


class TrainingData(TypedDict):
    match: List[RecordDictPair]
    distinct: List[RecordDictPair]


# Takes pairs of records and generates a (n_samples X n_features) array
FeaturizerFunction = Callable[
    [Sequence[RecordDictPair]], numpy.typing.NDArray[numpy.float_]
]


class Classifier(Protocol):
    """Takes an array of pairwise distances and computes the likelihood they are a pair."""

    def fit(self, X: numpy.typing.NDArray[numpy.float_], y: LabelsLike) -> None:
        ...

    def predict_proba(
        self, X: numpy.typing.NDArray[numpy.float_]
    ) -> numpy.typing.NDArray[numpy.float_]:
        ...


class ClosableJoinable(Protocol):
    def close(self) -> None:
        ...

    def join(self) -> None:
        ...


MapLike = Callable[[Callable[[Any], Any], Iterable], Iterable]

PathLike = Union[str, os.PathLike]
