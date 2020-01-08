import numpy

from typing import (Iterator,
                    Tuple,
                    Mapping,
                    Union,
                    Iterable,
                    List,
                    Any)

RecordDict = Mapping[str, Any]
RecordID = Union[int, str]
Record = Tuple[RecordID, RecordDict]
RecordPair = Tuple[Record, Record]
RecordPairs = Iterator[RecordPair]
Blocks = Iterator[RecordPairs]
Cluster = Tuple[Tuple[RecordID, ...], Union[numpy.ndarray, Tuple]]
Clusters = Iterable[Cluster]
Data = Mapping[RecordID, RecordDict]
TrainingExample = Tuple[RecordDict, RecordDict]
TrainingData = Mapping[str, List[TrainingExample]]
SearchResults = Iterable[Union[numpy.ndarray,
                               Tuple[Tuple[RecordID, RecordID], float]]]
LookupResults = Iterable[Tuple[RecordID, Tuple[Tuple[RecordID, float], ...]]]
