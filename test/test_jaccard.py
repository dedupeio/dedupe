import unittest
from dedupe.distance.jaccard import compareJaccard

class TestJaccard(unittest.TestCase):
    def setUp(self):
        self.set1 = 'a**b**c'
        self.set2 = 'c**d**e'
        self.set3 = 'x**y**z'

    def test_jaccard_equal(self):
        jaccard_val = compareJaccard(self.set1, self.set2)
        jaccard_val = round(jaccard_val, 5)
        self.assertEqual(jaccard_val, 0.2)

    def test_jaccard_zero(self):
        jaccard_zero = compareJaccard(self.set1, self.set3)
        self.assertEqual(jaccard_zero, 0.0)

        
if __name__ == '__main__':
    unittest.main()
