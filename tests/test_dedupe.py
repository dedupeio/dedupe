# -*- coding: utf-8 -*-

import unittest
import itertools

import numpy

import dedupe

DATA = {
    100: {"name": "Bob", "age": "50"},
    105: {"name": "Charlie", "age": "75"},
    110: {"name": "Meredith", "age": "40"},
    115: {"name": "Sue", "age": "10"},
    120: {"name": "Jimmy", "age": "20"},
    125: {"name": "Jimbo", "age": "21"},
    130: {"name": "Willy", "age": "35"},
    135: {"name": "William", "age": "35"},
    140: {"name": "Martha", "age": "19"},
    145: {"name": "Kyle", "age": "27"},
}

DATA_SAMPLE = (
    ({"age": "27", "name": "Kyle"}, {"age": "50", "name": "Bob"}),
    ({"age": "27", "name": "Kyle"}, {"age": "35", "name": "William"}),
    ({"age": "10", "name": "Sue"}, {"age": "35", "name": "William"}),
    ({"age": "27", "name": "Kyle"}, {"age": "20", "name": "Jimmy"}),
    ({"age": "75", "name": "Charlie"}, {"age": "21", "name": "Jimbo"}),
)


class DataModelTest(unittest.TestCase):
    def test_data_model(self):
        DataModel = dedupe.datamodel.DataModel

        self.assertRaises(TypeError, DataModel)

        data_model = DataModel(
            [
                {"field": "a", "variable name": "a", "type": "String"},
                {"field": "b", "variable name": "b", "type": "String"},
                {"type": "Interaction", "interaction variables": ["a", "b"]},
            ]
        )

        assert data_model._interaction_indices == [[0, 1]]

        data_model = DataModel(
            [
                {
                    "field": "a",
                    "variable name": "a",
                    "type": "String",
                    "has missing": True,
                },
                {"field": "b", "variable name": "b", "type": "String"},
                {"type": "Interaction", "interaction variables": ["a", "b"]},
            ]
        )

        assert data_model._missing_field_indices == [0, 2]

        data_model = DataModel(
            [
                {
                    "field": "a",
                    "variable name": "a",
                    "type": "String",
                    "has missing": False,
                },
                {"field": "b", "variable name": "b", "type": "String"},
                {"type": "Interaction", "interaction variables": ["a", "b"]},
            ]
        )

        assert data_model._missing_field_indices == []


class ConnectedComponentsTest(unittest.TestCase):
    def test_components(self):
        G = numpy.array(
            [
                ((1, 2), 0.1),
                ((2, 3), 0.2),
                ((4, 5), 0.2),
                ((4, 6), 0.2),
                ((7, 9), 0.2),
                ((8, 9), 0.2),
                ((10, 11), 0.2),
                ((12, 13), 0.2),
                ((12, 14), 0.5),
                ((11, 12), 0.2),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        components = dedupe.clustering.connected_components
        G_components = {
            frozenset(tuple(edge) for edge, _ in component)
            for component in components(G, 30000)
        }
        assert G_components == {
            frozenset(((1, 2), (2, 3))),
            frozenset(((4, 5), (4, 6))),
            frozenset(((12, 13), (12, 14), (10, 11), (11, 12))),
            frozenset(((7, 9), (8, 9))),
        }


class ClusteringTest(unittest.TestCase):
    def setUp(self):
        # Fully connected star network
        self.dupes = numpy.array(
            [
                ((1, 2), 0.86),
                ((1, 3), 0.72),
                ((1, 4), 0.2),
                ((1, 5), 0.6),
                ((2, 3), 0.86),
                ((2, 4), 0.2),
                ((2, 5), 0.72),
                ((3, 4), 0.3),
                ((3, 5), 0.5),
                ((4, 5), 0.72),
                ((10, 11), 0.9),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        # Dupes with Ids as String
        self.str_dupes = numpy.array(
            [
                (("1", "2"), 0.86),
                (("1", "3"), 0.72),
                (("1", "4"), 0.2),
                (("1", "5"), 0.6),
                (("2", "3"), 0.86),
                (("2", "4"), 0.2),
                (("2", "5"), 0.72),
                (("3", "4"), 0.3),
                (("3", "5"), 0.5),
                (("4", "5"), 0.72),
            ],
            dtype=[("pairs", "S4", 2), ("score", "f4")],
        )

        self.bipartite_dupes = (
            ((1, 5), 0.1),
            ((1, 6), 0.72),
            ((1, 7), 0.2),
            ((1, 8), 0.6),
            ((2, 5), 0.2),
            ((2, 6), 0.2),
            ((2, 7), 0.72),
            ((2, 8), 0.3),
            ((3, 5), 0.24),
            ((3, 6), 0.72),
            ((3, 7), 0.24),
            ((3, 8), 0.65),
            ((4, 5), 0.63),
            ((4, 6), 0.96),
            ((4, 7), 0.23),
            ((5, 8), 0.24),
        )

    def clusterEquals(self, x, y):
        if [] == x == y:
            return True
        if len(x) != len(y):
            return False

        for cluster_a, cluster_b in zip(x, y):
            if cluster_a[0] != cluster_b[0]:
                return False
            for score_a, score_b in zip(cluster_a[1], cluster_b[1]):
                if abs(score_a - score_b) > 0.001:
                    return False
            else:
                return True

    def test_hierarchical(self):
        hierarchical = dedupe.clustering.cluster
        assert self.clusterEquals(list(hierarchical(self.dupes, 1)), [])

        assert self.clusterEquals(
            list(hierarchical(self.dupes, 0.5)),
            [
                ((1, 2, 3), (0.778, 0.860, 0.778)),
                ((4, 5), (0.720, 0.720)),
                ((10, 11), (0.899, 0.899)),
            ],
        )

        print(hierarchical(self.dupes, 0.0))
        assert self.clusterEquals(
            list(hierarchical(self.dupes, 0)),
            [
                ((1, 2, 3, 4, 5), (0.526, 0.564, 0.542, 0.320, 0.623)),
                ((10, 11), (0.899, 0.899)),
            ],
        )

        assert list(hierarchical(self.str_dupes, 1)) == []
        assert list(zip(*hierarchical(self.str_dupes, 0.5)))[0] == (
            (b"1", b"2", b"3"),
            (b"4", b"5"),
        )
        assert list(zip(*hierarchical(self.str_dupes, 0)))[0] == (
            (b"1", b"2", b"3", b"4", b"5"),
        )

    def test_greedy_matching(self):
        greedyMatch = dedupe.clustering.greedyMatching

        bipartite_dupes = numpy.array(
            list(self.bipartite_dupes), dtype=[("ids", int, 2), ("score", float)]
        )

        assert list(greedyMatch(bipartite_dupes)) == [
            ((4, 6), 0.96),
            ((2, 7), 0.72),
            ((3, 8), 0.65),
            ((1, 5), 0.1),
        ]

    def test_gazette_matching(self):

        gazetteMatch = dedupe.clustering.gazetteMatching
        blocked_dupes = itertools.groupby(self.bipartite_dupes, key=lambda x: x[0][0])

        def to_numpy(x):
            return numpy.array(x, dtype=[("ids", int, 2), ("score", float)])

        blocked_dupes = [to_numpy(list(block)) for _, block in blocked_dupes]

        target = [
            (((1, 6), 0.72), ((1, 8), 0.6)),
            (((2, 7), 0.72), ((2, 8), 0.3)),
            (((3, 6), 0.72), ((3, 8), 0.65)),
            (((4, 6), 0.96), ((4, 5), 0.63)),
            (((5, 8), 0.24),),
        ]

        assert [
            tuple((tuple(pair), score) for pair, score in each.tolist())
            for each in gazetteMatch(blocked_dupes, n_matches=2)
        ] == target


class PredicatesTest(unittest.TestCase):
    def test_predicates_correctness(self):
        field = "123 16th st"
        assert dedupe.predicates.sortedAcronym(field) == ("11s",)
        assert dedupe.predicates.wholeFieldPredicate(field) == ("123 16th st",)
        assert dedupe.predicates.firstTokenPredicate(field) == ("123",)
        assert dedupe.predicates.firstTokenPredicate("") == ()
        assert dedupe.predicates.firstTokenPredicate("123/") == ("123",)
        assert dedupe.predicates.firstTwoTokensPredicate(field) == ("123 16th",)
        assert dedupe.predicates.firstTwoTokensPredicate("oneword") == ()
        assert dedupe.predicates.firstTwoTokensPredicate("") == ()
        assert dedupe.predicates.firstTwoTokensPredicate("123 456/") == ("123 456",)
        assert dedupe.predicates.tokenFieldPredicate(" ") == set([])
        assert dedupe.predicates.tokenFieldPredicate(field) == set(
            ["123", "16th", "st"]
        )
        assert dedupe.predicates.commonIntegerPredicate(field) == set(["123", "16"])
        assert dedupe.predicates.commonIntegerPredicate("foo") == set([])
        assert dedupe.predicates.firstIntegerPredicate("foo") == ()
        assert dedupe.predicates.firstIntegerPredicate("1foo") == ("1",)
        assert dedupe.predicates.firstIntegerPredicate("f1oo") == ()
        assert dedupe.predicates.sameThreeCharStartPredicate(field) == ("123",)
        assert dedupe.predicates.sameThreeCharStartPredicate("12") == ("12",)
        assert dedupe.predicates.commonFourGram("12") == set([])
        assert dedupe.predicates.sameFiveCharStartPredicate(field) == ("12316",)
        assert dedupe.predicates.sameSevenCharStartPredicate(field) == ("12316th",)
        assert dedupe.predicates.nearIntegersPredicate(field) == set(
            ["15", "17", "16", "122", "123", "124"]
        )
        assert dedupe.predicates.commonFourGram(field) == set(
            ["1231", "2316", "316t", "16th", "6ths", "thst"]
        )
        assert dedupe.predicates.commonSixGram(field) == set(
            ["12316t", "2316th", "316ths", "16thst"]
        )
        assert dedupe.predicates.initials(field, 12) == ("123 16th st",)
        assert dedupe.predicates.initials(field, 7) == ("123 16t",)
        assert dedupe.predicates.ngrams(field, 3) == [
            "123",
            "23 ",
            "3 1",
            " 16",
            "16t",
            "6th",
            "th ",
            "h s",
            " st",
        ]
        assert dedupe.predicates.commonTwoElementsPredicate((1, 2, 3)) == set(
            ("1 2", "2 3")
        )
        assert dedupe.predicates.commonTwoElementsPredicate((1,)) == set([])
        assert dedupe.predicates.commonThreeElementsPredicate((1, 2, 3)) == set(
            ("1 2 3",)
        )
        assert dedupe.predicates.commonThreeElementsPredicate((1,)) == set([])

        assert dedupe.predicates.fingerprint("time sandwich") == ("sandwichtime",)
        assert dedupe.predicates.oneGramFingerprint("sandwich time") == ("acdehimnstw",)
        assert dedupe.predicates.twoGramFingerprint("sandwich time") == (
            "anchdwhticimmendsatiwi",
        )
        assert dedupe.predicates.twoGramFingerprint("1") == ()
        assert dedupe.predicates.commonTwoTokens("foo bar") == set(["foo bar"])
        assert dedupe.predicates.commonTwoTokens("foo") == set([])


if __name__ == "__main__":
    unittest.main()
