#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from .canopy_index import CanopyIndex
from .index import Index
from .core import Enumerator

logger = logging.getLogger(__name__)


class TfIdfIndex(Index):
    def __init__(self):
        self._index = CanopyIndex()
        self._doc_to_id = Enumerator(start=1)
        self._parseTerms = self._index.lexicon.parseTerms

    def index(self, doc):
        if doc not in self._doc_to_id:
            i = self._doc_to_id[doc]
            self._index.index_doc(i, doc)

    def unindex(self, doc):
        i = self._doc_to_id.pop(doc)
        self._index.unindex_doc(i)
        self.initSearch()

    def initSearch(self):
        self._index.initSearch()

    def search(self, doc, threshold=0):
        query_list = self._parseTerms(doc)

        if query_list:
            results = [center for score, center
                       in self._index.apply(query_list, threshold)]
        else:
            results = []

        return results
