from typing import Iterable, Optional, Sequence, Type

from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity

from dedupe import predicates
from dedupe._typing import PredicateFunction
from dedupe.variables.base import FieldType, indexPredicates

crfEd = CRFEditDistance()

base_predicates = (
    predicates.wholeFieldPredicate,
    predicates.firstTokenPredicate,
    predicates.firstTwoTokensPredicate,
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
    predicates.sortedAcronym,
)


class BaseStringType(FieldType):
    _Predicate = predicates.StringPredicate
    _predicate_functions: Sequence[PredicateFunction] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.predicates += indexPredicates(
            (
                predicates.LevenshteinCanopyPredicate,
                predicates.LevenshteinSearchPredicate,
            ),
            (1, 2, 3, 4),
            self.field,
        )


class ShortStringType(BaseStringType):
    type = "ShortString"

    _predicate_functions = base_predicates + (
        predicates.commonFourGram,
        predicates.commonSixGram,
        predicates.tokenFieldPredicate,
        predicates.suffixArray,
        predicates.doubleMetaphone,
        predicates.metaphoneToken,
    )

    _index_predicates: Sequence[Type[predicates.IndexPredicate]] = [
        predicates.TfidfNGramCanopyPredicate,
        predicates.TfidfNGramSearchPredicate,
    ]
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(
        self, field: str, name: Optional[str] = None, crf: bool = False, **kwargs
    ):
        super().__init__(field, name=name, **kwargs)

        if crf:
            self.comparator = crfEd  # type: ignore[assignment]
        else:
            self.comparator = affineGap  # type: ignore[assignment]


class StringType(ShortStringType):
    type = "String"

    _index_predicates = [
        predicates.TfidfNGramCanopyPredicate,
        predicates.TfidfNGramSearchPredicate,
        predicates.TfidfTextCanopyPredicate,
        predicates.TfidfTextSearchPredicate,
    ]


class TextType(BaseStringType):
    type = "Text"

    _predicate_functions = base_predicates

    _index_predicates = [
        predicates.TfidfTextCanopyPredicate,
        predicates.TfidfTextSearchPredicate,
    ]
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, field: str, corpus: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(field, **kwargs)

        if corpus is None:
            corpus = []

        self.comparator = CosineTextSimilarity(corpus)  # type: ignore[assignment]
