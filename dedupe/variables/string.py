from .base import FieldType
from dedupe import predicates

from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity, CosineSetSimilarity

crfEd = CRFEditDistance()

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

class ShortStringType(FieldType) :
    type = "ShortString"

    _predicate_functions = (base_predicate_functions 
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


class TextType(FieldType) :
    type = "Text"

    _predicate_functions = base_predicate_functions 

    _index_predicates = (predicates.TfidfTextCanopyPredicate, 
                         predicates.TfidfTextSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(TextType, self).__init__(definition)


        if 'corpus' not in definition :
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
