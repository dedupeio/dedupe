import dedupe
import unittest
import numpy

class KFoldsTest(unittest.TestCase):

    def test_normal_k(self) :
        l = list(dedupe.crossvalidation.kFolds(numpy.array(range(6)), 6))
        assert len(l) == 6

    def test_small_k(self) :
        self.assertRaises(ValueError, 
                          lambda : list(dedupe.crossvalidation.kFolds(numpy.array(range(6)), 1)))

    def test_small_training(self) :
        self.assertRaises(ValueError, 
                          lambda : list(dedupe.crossvalidation.kFolds(numpy.array(range(1)), 2)))
        

    def test_large_k(self) :
        l = list(dedupe.crossvalidation.kFolds(numpy.array(range(2)), 10))

        assert len(l) == 2


if __name__ == "__main__":
    unittest.main()
