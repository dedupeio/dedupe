from .base import FieldType, Variable, indexPredicates
from dedupe import predicates

from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity, CosineSetSimilarity

crfEd = CRFEditDistance()

base_predicates = (predicates.wholeFieldPredicate,
                   predicates.firstTokenPredicate,
                   predicates.commonIntegerPredicate,
                   predicates.nearIntegersPredicate,
                   predicates.firstIntegerPredicate,
                   predicates.sameThreeCharStartPredicate,
                   predicates.sameFiveCharStartPredicate,
                   predicates.sameSevenCharStartPredicate,
                   predicates.commonTwoTokens,
                   predicates.commonThreeTokens,
                   predicates.fingerprint,
                   predicates.oneGramFingerprint,
                   predicates.twoGramFingerprint,
                   predicates.sortedAcronym)


class BaseStringType(FieldType) :
    type = None

    def __init__(self, definition) :
        self.field = definition['field']

        if 'variable name' in definition :
            self.name = definition['variable name'] 
        else :
            self.name = "(%s: %s)" % (self.field, self.type)

        self.predicates = [predicates.StringPredicate(pred, self.field) 
                           for pred in self._predicate_functions]

        self.predicates += indexPredicates(self._index_predicates,
                                           self._index_thresholds,
                                           self.field)

        Variable.__init__(self, definition)

    


class ShortStringType(BaseStringType) :
    type = "ShortString"

    _predicate_functions = (base_predicates 
                            + (predicates.commonFourGram,
                               predicates.commonSixGram,
                               predicates.tokenFieldPredicate,
                               predicates.suffixArray,
                               predicates.doubleMetaphone,
                               predicates.metaphoneToken))

    _index_predicates = (predicates.TfidfNGramCanopyPredicate, 
                         predicates.TfidfNGramSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)


    def __init__(self, definition) :
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) == True :
            self.comparator = crfEd
        else :
            self.comparator = affineGap

class StringType(ShortStringType) :
    type = "String"

    _index_predicates = (predicates.TfidfNGramCanopyPredicate, 
                         predicates.TfidfNGramSearchPredicate,
                         predicates.TfidfTextCanopyPredicate, 
                         predicates.TfidfTextSearchPredicate)


class TextType(BaseStringType) :
    type = "Text"

    _predicate_functions = base_predicates 

    _index_predicates = (predicates.TfidfTextCanopyPredicate, 
                         predicates.TfidfTextSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(TextType, self).__init__(definition)

        if 'corpus' not in definition :
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
