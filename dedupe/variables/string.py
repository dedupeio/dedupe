from .base import FieldType
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
                   predicates.sortedAcronym,
                   predicates.tokenFieldPredicate,
                   predicates.commonFourGram,
                   predicates.commonSixGram,
                   predicates.hundredIntegerPredicate,
                   predicates.hundredIntegersOddPredicate,
                   predicates.nearIntegersPredicate,
                   predicates.alphaNumericPredicate,
                   predicates.commonIntegerPredicate,
                   predicates.suffixArray,
                   predicates.metaphoneToken)


class BaseStringType(FieldType):
    type = None
    _Predicate = predicates.StringPredicate
    _OverlapPredicate = predicates.MinHashStringPredicate

    def __init__(self, definition):
        super(BaseStringType, self).__init__(definition)

        self.predicates += self.indexPredicates((predicates.LevenshteinCanopyPredicate,
                                                 predicates.LevenshteinSearchPredicate),
                                                (1, 2, 3, 4),
                                                self.field)


class ShortStringType(BaseStringType):
    type = "ShortString"

    _predicate_functions = (base_predicates +
                            (predicates.doubleMetaphone,))

    _index_predicates = (predicates.TfidfNGramCanopyPredicate,
                         predicates.TfidfNGramSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    _overlap_predicates = (predicates.tokenFieldPredicate,
                           predicates.commonFourGram,
                           predicates.commonSixGram,
                           predicates.hundredIntegerPredicate,
                           predicates.hundredIntegersOddPredicate,
                           predicates.nearIntegersPredicate,
                           predicates.alphaNumericPredicate,
                           predicates.commonIntegerPredicate,
                           predicates.suffixArray,
                           predicates.metaphoneToken)
    _overlap_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) is True:
            self.comparator = crfEd
        else:
            self.comparator = affineGap

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
