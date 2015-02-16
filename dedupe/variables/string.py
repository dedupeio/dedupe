from base import FieldType
from dedupe import predicates

from affinegap import normalizedAffineGapDistance
from simplecosine.cosine import CosineTextSimilarity, CosineSetSimilarity

class ShortStringType(FieldType) :
    comparator = normalizedAffineGapDistance
    type = "ShortString"

    _predicate_functions = (predicates.wholeFieldPredicate,
                            predicates.tokenFieldPredicate,
                            predicates.firstTokenPredicate,
                            predicates.commonIntegerPredicate,
                            predicates.nearIntegersPredicate,
                            predicates.firstIntegerPredicate,
                            predicates.sameThreeCharStartPredicate,
                            predicates.sameFiveCharStartPredicate,
                            predicates.sameSevenCharStartPredicate,
                            predicates.commonFourGram,
                            predicates.commonSixGram,
                            predicates.suffixArray,
                            predicates.commonTwoTokens,
                            predicates.commonThreeTokens,
                            predicates.fingerprint,
                            predicates.oneGramFingerprint,
                            predicates.twoGramFingerprint,
                            predicates.sortedAcronym,
                            predicates.doubleMetaphone,
                            predicates.metaphoneToken)

class StringType(ShortStringType) :
    comparator = normalizedAffineGapDistance
    type = "String"

    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(StringType, self).__init__(definition)

        self.predicates += [predicates.TfidfTextPredicate(threshold, 
                                                          self.field)
                            for threshold in self._canopy_thresholds]

        self.predicates += [predicates.TfidfNGramPredicate(threshold, 
                                                          self.field)
                            for threshold in self._canopy_thresholds]



class TextType(StringType) :
    type = "Text"

    def __init__(self, definition) :
        super(TextType, self).__init__(definition)

        if 'corpus' not in definition :
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
