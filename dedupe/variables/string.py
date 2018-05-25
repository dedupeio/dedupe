from .base import FieldType, indexPredicates
from dedupe import predicates

from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity

crfEd = CRFEditDistance()

base_predicates = (predicates.wholeFieldPredicate,
                   predicates.firstTokenPredicate,
                   predicates.firstIntegerPredicate,
                   predicates.sameThreeCharStartPredicate,
                   predicates.sameFiveCharStartPredicate,
                   predicates.sameSevenCharStartPredicate,
                   predicates.fingerprint,
                   predicates.oneGramFingerprint,
                   predicates.twoGramFingerprint,
                   predicates.sortedAcronym)


class BaseStringType(FieldType):
    type = None
    _Predicate = predicates.StringPredicate

    def __init__(self, definition):
        super(BaseStringType, self).__init__(definition)

        self.predicates += indexPredicates((predicates.LevenshteinCanopyPredicate,
                                            predicates.LevenshteinSearchPredicate),
                                           (1, 2, 3, 4),
                                           self.field)


class ShortStringType(BaseStringType):
    type = "ShortString"

    _predicate_functions = (base_predicates +
                            (predicates.suffixArray,
                             predicates.doubleMetaphone))

    _index_predicates = (predicates.TfidfNGramCanopyPredicate,
                         predicates.TfidfNGramSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) is True:
            self.comparator = crfEd
        else:
            self.comparator = affineGap

        overlap_preds = [predicates.tokenFieldPredicate,
                         predicates.commonFourGram,
                         predicates.commonSixGram,
                         predicates.hundredIntegerPredicate,
                         predicates.hundredIntegersOddPredicate,
                         predicates.nearIntegersPredicate,
                         predicates.alphaNumericPredicate,
                         predicates.commonIntegerPredicate,
                         predicates.metaphoneToken]
                             
        for n_common in range(1, 4):
            for pred in overlap_preds:
                self.predicates.append(self._Predicate(pred,
                                                       self.field,
                                                       n_common))

        

class StringType(ShortStringType):
    type = "String"

    _index_predicates = (predicates.TfidfNGramCanopyPredicate,
                         predicates.TfidfNGramSearchPredicate,
                         predicates.TfidfTextCanopyPredicate,
                         predicates.TfidfTextSearchPredicate)


class TextType(BaseStringType):
    type = "Text"

    _predicate_functions = base_predicates

    _index_predicates = (predicates.TfidfTextCanopyPredicate,
                         predicates.TfidfTextSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(TextType, self).__init__(definition)

        if 'corpus' not in definition:
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
