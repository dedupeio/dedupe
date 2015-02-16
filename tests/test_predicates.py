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

    def test_first_last(self) :
        block_val = predicates.lastSetElementPredicate(self.s1)
        assert block_val == ('red',)
        block_val = predicates.firstSetElementPredicate(self.s1)
        assert block_val == ('blue',)
        block_val = predicates.firstSetElementPredicate(set([]))
        assert block_val == ()
        block_val = predicates.lastSetElementPredicate(set([]))
        assert block_val == ()

class TestLatLongGrid(unittest.TestCase):
    def setUp(self):
        self.latlong1 = (42.535, -5.012)

    def test_precise_latlong(self):
        block_val = predicates.latLongGridPredicate(self.latlong1)
        assert block_val == (u'[42.5, -5.0]',)
        block_val = predicates.latLongGridPredicate((0,0))
        assert block_val == ()

    def test_exists(self) :
        block_val = predicates.existsPredicate(self.latlong1)
        assert block_val == (u'1',)
        block_val = predicates.existsPredicate((0,0))
        print block_val
        assert block_val == (u'0',)

class TestNumericPredicates(unittest.TestCase) :
    def test_order_of_magnitude(self) :
        assert predicates.orderOfMagnitude(10) == (u'1',)
        assert predicates.orderOfMagnitude(9) == (u'1',)
        assert predicates.orderOfMagnitude(2) == (u'0',)


    def test_round_to_1(self) :
        assert predicates.roundTo1(22315) == (u'20000',)

if __name__ == '__main__':
    unittest.main()

