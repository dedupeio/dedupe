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
        l = list(dedupe.crossvalidation.kFolds(numpy.array(range(4)), 10))

        assert len(l) == 4

class scoreTest(unittest.TestCase) :

    def test_no_true(self) :
        score = dedupe.crossvalidation.scorePredictions(numpy.zeros(5), 
                                                        numpy.ones(5))
        assert score == 0

    def test_no_predicted(self) :
        score = dedupe.crossvalidation.scorePredictions(numpy.ones(5), 
                                                        numpy.zeros(5))
        assert score == 0

    def test_all_predicted(self) :
        score = dedupe.crossvalidation.scorePredictions(numpy.ones(5), 
                                                        numpy.ones(5))
        assert score == 1

    def test_all_predicted(self) :
        score = dedupe.crossvalidation.scorePredictions(numpy.array([1,0,1,0]), 
                                                        numpy.array([1,1,0,0]))
        assert score == 0


class scoreReduction(unittest.TestCase) :
    def test_nones(self) :
        avg_score = dedupe.crossvalidation.reduceScores([None, None])
        assert avg_score == 0

    def test_some_nones(self) :
        avg_score = dedupe.crossvalidation.reduceScores([1, None])
        assert avg_score == 1

    def test_no_nones(self) :
        avg_score = dedupe.crossvalidation.reduceScores([1, 0])
        assert avg_score == 0.5




if __name__ == "__main__":
    unittest.main()
