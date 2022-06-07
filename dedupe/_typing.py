import sys
from typing import (
    Any,
    Callable,
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


RecordDict = Mapping[str, Any]
RecordID = Union[int, str]
RecordIDDType = Union[Type[int], tuple[Type[str], Literal[256]]]
Record = Tuple[RecordID, RecordDict]
RecordPair = Tuple[Record, Record]
RecordPairs = Iterator[RecordPair]
Block = List[RecordPair]
Blocks = Iterator[Block]
Cluster = Tuple[
    Tuple[RecordID, ...], Union[numpy.typing.NDArray[numpy.float_], Tuple[float, ...]]
]
Clusters = Iterable[Cluster]
Data = Mapping[RecordID, RecordDict]
TrainingExample = Tuple[RecordDict, RecordDict]
TrainingExamples = List[TrainingExample]
Links = Iterable[Union[numpy.ndarray, Tuple[Tuple[RecordID, RecordID], float]]]
LookupResults = Iterable[Tuple[RecordID, Tuple[Tuple[RecordID, float], ...]]]
JoinConstraint = Literal["one-to-one", "many-to-one", "many-to-many"]
Comparator = Callable[[Any, Any], Union[Union[int, float], Sequence[Union[int, float]]]]
Scores = Union[numpy.memmap, numpy.ndarray]
Labels = List[Literal[0, 1]]
LabelsLike = Iterable[Literal[0, 1]]

VariableDefinition = TypedDict(
    "VariableDefinition",
    {
        "type": str,
        "field": str,
        "variable name": str,
        "corpus": Iterable[Union[str, Sequence[str]]],
        "comparator": Callable[
            [Any, Any], Union[int, float]
        ],  # a custom comparator can only return a single float or int, not a sequence of numbers
        "categories": List[str],
        "interaction variables": List[str],
        "has missing": bool,
    },
    total=False,
)


class TrainingData(TypedDict):
    match: List[TrainingExample]
    distinct: List[TrainingExample]


class Classifier(Protocol):
    def fit(self, X: object, y: object) -> None:
        ...

    def predict_proba(self, X: object) -> numpy.typing.NDArray[numpy.float_]:
        ...

class ClosableJoinable(Protocol):
    
    def close(self):
        ...
    
    def join(self):
        ...

MapLike = Callable[[Callable, Iterable], Iterable]
    
