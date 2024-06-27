import unittest

from dedupe import predicates


class TestPuncStrip(unittest.TestCase):
    def test_sevenchar(self):
        s1 = predicates.StringPredicate(predicates.sameSevenCharStartPredicate, "foo")
        assert s1({"foo": "fo,18v*1vaad80"}) == s1({"foo": "fo18v1vaad80"})

    def test_set(self):
        s1 = predicates.SimplePredicate(predicates.wholeSetPredicate, "foo")
        colors = {"red", "blue", "green"}
        assert s1({"foo": colors}) == {str(colors)}


class TestMetaphone(unittest.TestCase):
    def test_metaphone_token(self):
        block_val = predicates.metaphoneToken("9301 S. State St. ")
        assert block_val == {"STT", "S", "ST"}


class TestWholeSet(unittest.TestCase):
    def setUp(self):
        self.s1 = {"red", "blue", "green"}

    def test_full_set(self):
        block_val = predicates.wholeSetPredicate(self.s1)
        self.assertEqual(block_val, {str(self.s1)})


class TestSetElement(unittest.TestCase):
    def setUp(self):
        self.s1 = {"red", "blue", "green"}

    def test_long_set(self):
        block_val = predicates.commonSetElementPredicate(self.s1)
        self.assertEqual(set(block_val), {"blue", "green", "red"})

    def test_empty_set(self):
        block_val = predicates.commonSetElementPredicate(set())
        self.assertEqual(block_val, set())

    def test_first_last(self):
        block_val = predicates.lastSetElementPredicate(self.s1)
        assert block_val == {"red"}
        block_val = predicates.firstSetElementPredicate(self.s1)
        assert block_val == {"blue"}

    def test_magnitude(self):
        block_val = predicates.magnitudeOfCardinality(self.s1)
        assert block_val == {"0"}

        block_val = predicates.magnitudeOfCardinality(())
        assert block_val == set()


class TestLatLongGrid(unittest.TestCase):
    def setUp(self):
        self.latlong1 = (42.535, -5.012)

    def test_precise_latlong(self):
        block_val = predicates.latLongGridPredicate(self.latlong1)
        assert block_val == {"(42.5, -5.0)"}
        block_val = predicates.latLongGridPredicate((0, 0))
        assert block_val == set()


class TestAlpaNumeric(unittest.TestCase):
    def test_alphanumeric(self):
        assert predicates.alphaNumericPredicate("a1") == {"a1"}
        assert predicates.alphaNumericPredicate("1a") == {"1a"}
        assert predicates.alphaNumericPredicate("a1b") == {"a1b"}
        assert predicates.alphaNumericPredicate("1 a") == {"1"}
        assert predicates.alphaNumericPredicate("a1 b1") == {"a1", "b1"}
        assert predicates.alphaNumericPredicate("asdf") == set()
        assert predicates.alphaNumericPredicate("1") == {"1"}
        assert predicates.alphaNumericPredicate("a_1") == {"1"}
        assert predicates.alphaNumericPredicate("a$1") == {"1"}
        assert predicates.alphaNumericPredicate("a 1") == {"1"}
        assert predicates.alphaNumericPredicate("773-555-1676") == {
            "773",
            "555",
            "1676",
        }


class TestNumericPredicates(unittest.TestCase):
    def test_order_of_magnitude(self):
        assert predicates.orderOfMagnitude(10) == {"1"}
        assert predicates.orderOfMagnitude(9) == {"1"}
        assert predicates.orderOfMagnitude(2) == {"0"}
        assert predicates.orderOfMagnitude(-2) == set()

    def test_round_to_1(self):
        assert predicates.roundTo1(22315) == {"20000"}
        assert predicates.roundTo1(-22315) == {"-20000"}


class TestCompoundPredicate(unittest.TestCase):
    def test_escapes_colon(self):
        """
        Regression test for issue #836
        """
        predicate_1 = predicates.SimplePredicate(
            predicates.commonSetElementPredicate, "col_1"
        )
        predicate_2 = predicates.SimplePredicate(
            predicates.commonSetElementPredicate, "col_2"
        )
        record = {"col_1": ["foo:", "foo"], "col_2": [":bar", "bar"]}

        block_val = predicates.CompoundPredicate([predicate_1, predicate_2])(record)
        assert len(set(block_val)) == 4
        assert block_val == {"foo\\::\\:bar", "foo\\::bar", "foo:\\:bar", "foo:bar"}

    def test_escapes_escaped_colon(self):
        """
        Regression test for issue #836
        """
        predicate_1 = predicates.SimplePredicate(
            predicates.commonSetElementPredicate, "col_1"
        )
        predicate_2 = predicates.SimplePredicate(
            predicates.commonSetElementPredicate, "col_2"
        )
        record = {"col_1": ["foo\\:", "foo"], "col_2": ["\\:bar", "bar"]}

        block_val = predicates.CompoundPredicate([predicate_1, predicate_2])(record)
        assert len(set(block_val)) == 4
        assert block_val == {
            "foo\\\\::\\\\:bar",
            "foo\\\\::bar",
            "foo:\\\\:bar",
            "foo:bar",
        }


if __name__ == "__main__":
    unittest.main()
