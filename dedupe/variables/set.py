from typing import Collection, Iterable, Optional

from simplecosine.cosine import CosineSetSimilarity

from dedupe import predicates
from dedupe.variables.base import FieldType


class SetType(FieldType):
    type = "Set"

    _predicate_functions = (
        predicates.wholeSetPredicate,
        predicates.commonSetElementPredicate,
        predicates.lastSetElementPredicate,
        predicates.commonTwoElementsPredicate,
        predicates.commonThreeElementsPredicate,
        predicates.magnitudeOfCardinality,
        predicates.firstSetElementPredicate,
    )

    _index_predicates = (
        predicates.TfidfSetSearchPredicate,
        predicates.TfidfSetCanopyPredicate,
    )
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(
        self, field: str, corpus: Optional[Iterable[Collection[str]]] = None, **kwargs
    ):
        super().__init__(field, **kwargs)

        if corpus is None:
            corpus = []

        self.comparator = CosineSetSimilarity(corpus)  # type: ignore[assignment]
