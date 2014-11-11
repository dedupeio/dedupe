import unittest
from dedupe import predicates

class TestWholeSet(unittest.TestCase):
    def setUp(self):
        self.s1 = set(['red', 'blue', 'green'])

    def test_full_set(self):
        block_val = predicates.wholeSetPredicate(self.s1)
        self.assertEqual(block_val, (unicode(self.s1),))

    def test_empty_set(self):
        block_val = predicates.wholeSetPredicate(set())
        self.assertEqual(block_val, tuple())

class TestSetElement(unittest.TestCase):
    def setUp(self):
        self.s1 = set(['red', 'blue', 'green'])

    def test_long_set(self):
        block_val = predicates.commonSetElementPredicate(self.s1)
        self.assertEqual(block_val, ('blue', 'green', 'red'))

    def test_empty_set(self):
        block_val = predicates.commonSetElementPredicate(set())
        self.assertEqual(block_val, tuple())

class TestLatLongGrid(unittest.TestCase):
    def setUp(self):
        self.latlong1 = (42.535, -5.012)

    def test_precise_latlong(self):
        block_val = predicates.latLongGridPredicate(self.latlong1)
        print block_val
        assert block_val == (u'[42.5, -5.0]',)

if __name__ == '__main__':
    unittest.main()

