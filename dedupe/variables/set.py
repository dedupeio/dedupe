from .base import FieldType
from dedupe import predicates
from simplecosine.cosine import CosineSetSimilarity


class SetType(FieldType):
    type = "Set"

    _predicate_functions = (predicates.wholeSetPredicate,
                            predicates.lastSetElementPredicate,
                            predicates.magnitudeOfCardinality,
                            predicates.firstSetElementPredicate)

    _index_predicates = (predicates.TfidfSetSearchPredicate,
                         predicates.TfidfSetCanopyPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    _overlap_predicates = (predicates.commonSetElementPredicate,)
    _overlap_thresholds = (1, 2, 3, 4)

    def __init__(self, definition):
        super(SetType, self).__init__(definition)

        if 'corpus' not in definition:
            definition['corpus'] = []

        self.comparator = CosineSetSimilarity(definition['corpus'])
