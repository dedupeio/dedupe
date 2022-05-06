from .base import FieldType
from dedupe import predicates

import sklearn.feature_extraction.text
import sklearn.metrics.pairwise


def no_op(x):
    return x


class TfidfSetVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        return no_op


class SetType(FieldType):
    type = "Set"

    _predicate_functions = (
        predicates.wholeSetPredicate,
        predicates.commonSetElementPredicate,
        predicates.lastSetElementPredicate,
        predicates.commonTwoElementsPredicate,
        predicates.commonThreeElementsPredicate,
        predicates.magnitudeOfCardinality,
        predicates.firstSetElementPredicate,
    )

    _index_predicates = (
        predicates.TfidfSetSearchPredicate,
        predicates.TfidfSetCanopyPredicate,
    )
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(SetType, self).__init__(definition)

        if "corpus" not in definition:
            definition["corpus"] = []

        corpus = (doc for doc in definition["corpus"] if doc)

        self.vectorizer = TfidfSetVectorizer()
        self.vectorizer.fit(corpus)

        self._cosine = sklearn.metrics.pairwise.cosine_similarity

    def comparator(self, field_1, field_2):

        return self._cosine(
            self.vectorizer.transform([field_1]), self.vectorizer.transform([field_2])
        )
