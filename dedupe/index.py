from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import MutableMapping, Tuple

    Doc = Tuple[str, ...]


class Index(ABC):
    _doc_to_id: MutableMapping[Doc, int]

    @abstractmethod
    def __init__(self) -> None:  # pragma: no cover
        pass

    @abstractmethod
    def index(self, doc: Doc) -> None:  # pragma: no cover
        pass

    @abstractmethod
    def unindex(self, doc: Doc) -> None:  # pragma: no cover
        pass

    @abstractmethod  # pragma: no cover
    def search(self, doc: Doc, threshold: int | float = 0) -> list[int]:
        pass

    @abstractmethod
    def initSearch(self) -> None:  # pragma: no cover
        pass
