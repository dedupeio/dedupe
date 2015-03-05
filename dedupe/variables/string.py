from base import FieldType
from dedupe import predicates

from affinegap import normalizedAffineGapDistance
from simplecosine.cosine import CosineTextSimilarity, CosineSetSimilarity

base_predicate_functions = (predicates.wholeFieldPredicate,
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

class IndexNGramString(object) :

    def __init__(self, definition) :
        super(IndexNGramString, self).__init__(definition)

        self.predicates += [predicates.TfidfNGramPredicate(threshold, 
                                                          self.field)
                            for threshold in self._canopy_thresholds]


class IndexTextString(object) :

    def __init__(self, definition) :
        super(IndexTextString, self).__init__(definition)

        self.predicates += [predicates.TfidfTextPredicate(threshold, 
                                                          self.field)
                            for threshold in self._canopy_thresholds]




class ShortStringType(IndexNGramString, FieldType) :
    comparator = normalizedAffineGapDistance
    type = "ShortString"

    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    _predicate_functions = (base_predicate_functions 
                            + (predicates.commonFourGram,
                               predicates.commonSixGram,
                               predicates.tokenFieldPredicate,
                               predicates.suffixArray,
                               predicates.doubleMetaphone,
                               predicates.metaphoneToken))


class StringType(IndexTextString, ShortStringType) :
    comparator = normalizedAffineGapDistance
    type = "String"



class TextType(IndexTextString, FieldType) :
    type = "Text"

    _predicate_functions = base_predicate_functions 
    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(TextType, self).__init__(definition)


        if 'corpus' not in definition :
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
