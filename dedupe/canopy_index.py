from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy
from BTrees.Length import Length
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.setops import mass_weightedUnion
from zope.index.text.textindex import TextIndex

logger = logging.getLogger(__name__)


class CanopyIndex(TextIndex):  # pragma: no cover
    def __init__(self) -> None:
        lexicon = CanopyLexicon()
        self.index = CosineIndex(lexicon)
        self.lexicon = lexicon

    def initSearch(self) -> None:
        N = len(self.index._docweight)
        threshold = int(max(1000, N * 0.05))

        stop_words = []
        self._wids_dict = {}

        bucket = self.index.family.IF.Bucket
        for wid, docs in self.index._wordinfo.items():
            if len(docs) > threshold:
                stop_words.append(wid)
                continue

            if isinstance(docs, dict):
                docs = bucket(docs)
                self.index._wordinfo[wid] = docs

            idf = numpy.log1p(N / len(docs))
            term = self.lexicon._words[wid]

            self._wids_dict[term] = (wid, idf)

        for wid in stop_words:
            word = self.lexicon._words.pop(wid)
            del self.lexicon._wids[word]
            logger.info(f"Removing stop word {word}")
            del self.index._wordinfo[wid]

    def apply(
        self,
        query_list: Iterable[str],
        threshold: float,
        start: int = 0,
        count: int | None = None,
    ) -> list[tuple[float, int]]:
        _wids_dict = self._wids_dict
        _wordinfo = self.index._wordinfo
        l_pow = float.__pow__

        L = []
        qw = 0.0

        for term in query_list:
            wid, weight = _wids_dict.get(term, (None, None))
            if wid is None:
                continue
            docs = _wordinfo[wid]
            L.append((docs, weight))
            qw += l_pow(weight, 2)

        results = mass_weightedUnion(L)

        qw = math.sqrt(qw)
        filtered_results: list[tuple[float, int]] = results.byValue(qw * threshold)

        return filtered_results


class CanopyLexicon(Lexicon):  # pragma: no cover
    def sourceToWordIds(self, last: list | None = None) -> list[int]:
        if last is None:
            last = []
        if not isinstance(self.wordCount, Length):  # type: ignore[has-type]
            self.wordCount = Length(self.wordCount())  # type: ignore[has-type]
        self.wordCount._p_deactivate()
        return list(map(self._getWordIdCreate, last))
