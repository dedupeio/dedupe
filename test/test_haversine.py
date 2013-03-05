import unittest
from dedupe.haversine import compareLatLong

class TestHaversine(unittest.TestCase):
    def setUp(self):
        self.sfo = '37.619105**-122.375236'
        self.ord = '41.981649**-87.906670'

    def test_haversine_equal(self):
        km_dist_val = compareLatLong(self.sfo, self.ord)

        self.assertAlmostEqual(km_dist_val, 2964, -1)

    def test_haversine_zero(self):
        km_dist_zero = compareLatLong(self.ord, self.ord)
        self.assertAlmostEqual(km_dist_zero, 0.0, 0)

        
if __name__ == '__main__':
    unittest.main()
