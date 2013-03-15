import unittest
from dedupe import predicates

class TestCommonSet(unittest.TestCase):
    def setUp(self):
        self.s1 = set('red', 'blue', 'green')

    def test_full_set(self):
        block_val = predicate.commonSetPredicate(self.s1)
        self.assertEqual(block_val, str(s1))

    def test_empty_set(self):
        block_val = predicate.commonSetPredicate(set())
        self.assertEqual(block_val, tuple())

class TestSetElement(unittest.TestCase):
    def setUp(self):
        self.s1 = set('red', 'blue', 'green')

    def test_long_set(self):
        block_val = predicate.commonSetElementPredicate(self.s1)
        self.assertEqual(block_val, ('red', 'blue', 'green'))

    def test_empty_set(self):
        block_val = predicate.commonSetElementPredicate(set())
        self.assertEqual(block_val, tuple())

class TestLatLongGrid(unittest.TestCase):
    def setUp(self):
        self.latlong1 = (42.335, -5.212)
        self.latlong1grid = str( ((43.3, 43.4),(-5.2, -5.3)))
        self.latlong2 = (42, -5)
        self.latlong2grid = str(((41.9, 42.1), (-5.1, -4.9)))

    def test_precise_latlong(self):
        block_val = predicates.latLongGridPredicate(self.latlong1)
        self.assertEqual(self.latlong1grid, block_val)
        
    def test_imprecise_latlong(self):
        block_val = predicates.latLongGridPredicate(self.latlong2)
        self.assertEqual(self.latlong2grid, block_val)

if __name__ == '__main__':
    unittest.main()

