import numpy
import sys

from typing import (Iterator,
                    Tuple,
                    Mapping,
                    Union,
                    Iterable,
                    List,
                    Any)

if sys.version_info >= (3, 8):
    from typing import TypedDict, Protocol, Literal
else:
    from typing_extensions import TypedDict, Protocol, Literal


RecordDict = Mapping[str, Any]
RecordID = Union[int, str]
Record = Tuple[RecordID, RecordDict]
RecordPair = Tuple[Record, Record]
RecordPairs = Iterator[RecordPair]
Blocks = Iterator[List[RecordPair]]
Cluster = Tuple[Tuple[RecordID, ...], Union[numpy.ndarray, Tuple]]
Clusters = Iterable[Cluster]
Data = Mapping[RecordID, RecordDict]
TrainingExample = Tuple[RecordDict, RecordDict]
Links = Iterable[Union[numpy.ndarray,
                       Tuple[Tuple[RecordID, RecordID], float]]]
LookupResults = Iterable[Tuple[RecordID, Tuple[Tuple[RecordID, float], ...]]]
JoinConstraint = Literal['one-to-one', 'many-to-one', 'many-to-many']


class TrainingData(TypedDict):
    match: List[TrainingExample]
    distinct: List[TrainingExample]


class Classifier(Protocol):
    def fit(self, X: object, y: object) -> None:
        ...

    def predict_proba(self, X: object) -> Any:
        ...
