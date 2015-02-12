from base import FieldType
from dedupe import predicates
from simplecosine.cosine import CosineSetSimilarity

class SetType(FieldType) :
    type = "Set"

    _predicate_functions = (predicates.wholeSetPredicate,
                            predicates.commonSetElementPredicate,
                            predicates.lastSetElementPredicate,
                            predicates.commonTwoElementsPredicate,
                            predicates.commonThreeElementsPredicate,
                            predicates.firstSetElementPredicate)
    
    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(SetType, self).__init__(definition)

        canopy_predicates = [predicates.TfidfPredicate(threshold, 
                                                       self.field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

        if 'corpus' not in definition :
            definition['corpus'] = [] 

        self.comparator = CosineSetSimilarity(definition['corpus'])


