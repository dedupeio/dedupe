from .base import FieldType, indexPredicates
from dedupe import predicates
import numpy as np
from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity

crfEd = CRFEditDistance()

base_predicates = (predicates.wholeFieldPredicate,
                   predicates.firstTokenPredicate,
                   predicates.commonIntegerPredicate,
                   predicates.nearIntegersPredicate,
                   predicates.firstIntegerPredicate,
                   predicates.hundredIntegerPredicate,
                   predicates.hundredIntegersOddPredicate,
                   predicates.alphaNumericPredicate,
                   predicates.sameThreeCharStartPredicate,
                   predicates.sameFiveCharStartPredicate,
                   predicates.sameSevenCharStartPredicate,
                   predicates.commonTwoTokens,
                   predicates.commonThreeTokens,
                   predicates.fingerprint,
                   predicates.oneGramFingerprint,
                   predicates.twoGramFingerprint,
                   predicates.sortedAcronym)


def string_comparator(field_1, field_2):
    affine_distance = affineGap(field_1, field_2)
    x = np.exp(0.5*affine_distance)-np.exp(1)/2
    x0 = 5
    a = 1
    return np.exp(a*x-x0)/(np.exp(a*x-x0)+1)


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
                            (predicates.commonFourGram,
                             predicates.commonSixGram,
                             predicates.tokenFieldPredicate,
                             predicates.suffixArray,
                             predicates.doubleMetaphone,
                             predicates.metaphoneToken))

    _index_predicates = (predicates.TfidfNGramCanopyPredicate,
                         predicates.TfidfNGramSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) is True:
            self.comparator = crfEd
        else:
            self.comparator = string_comparator


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
