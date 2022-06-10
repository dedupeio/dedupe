from typing import Dict, List

import Levenshtein_search

from .core import Enumerator
from .index import Index


class LevenshteinIndex(Index):
    _doc_to_id: Dict[str, int]  # type: ignore[assignment]

    def __init__(self) -> None:
        self.index_key = Levenshtein_search.populate_wordset(-1, [])
        self._doc_to_id = Enumerator(start=1)

    def index(self, doc: str) -> None:  # type: ignore[override]
        if doc not in self._doc_to_id:
            self._doc_to_id[doc]
            Levenshtein_search.add_string(self.index_key, doc)

    def unindex(self, doc: str) -> None:  # type: ignore[override]
        del self._doc_to_id[doc]
        Levenshtein_search.clear_wordset(self.index_key)
        self.index_key = Levenshtein_search.populate_wordset(-1, list(self._doc_to_id))

    def initSearch(self) -> None:
        pass

    def search(self, doc: str, threshold: int = 0) -> List[int]:  # type: ignore[override]
        matching_docs = Levenshtein_search.lookup(self.index_key, doc, threshold)
        if matching_docs:
            return [self._doc_to_id[match] for match, _, _ in matching_docs]
        else:
            return []

    def __del__(self) -> None:
        Levenshtein_search.clear_wordset(self.index_key)
