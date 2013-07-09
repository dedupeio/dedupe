import unittest
from dedupe.distance.haversine import compareLatLong
import numpy

class TestHaversine(unittest.TestCase):
    def setUp(self):
        self.sfo = (37.619105, -122.375236)
        self.ord = (41.981649, -87.906670)

    def test_haversine_equal(self):
        km_dist_val = compareLatLong(self.sfo, self.ord)

        self.assertAlmostEqual(km_dist_val, 2964, -1)

    def test_haversine_zero(self):
        km_dist_zero = compareLatLong(self.ord, self.ord)
        self.assertAlmostEqual(km_dist_zero, 0.0, 0)

    def test_haversine_na(self):
        km_dist_na = compareLatLong((0.0, 0.0), (1.0, 2.0))
        assert numpy.isnan(km_dist_na)
        km_dist_na = compareLatLong((1.0, 2.0), (0.0, 0.0))
        assert numpy.isnan(km_dist_na)
        km_dist_n_na = compareLatLong((0.0, 1.0), (1.0, 2.0))
        self.assertAlmostEqual(km_dist_n_na, 157, -1)


        
if __name__ == '__main__':
    unittest.main()
