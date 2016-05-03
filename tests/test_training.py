import dedupe
import unittest
try:
    import unittest.mock as mock
except ImportError:
    import mock
import numpy


class UndertainTest(unittest.TestCase) :
    def setUp(self) :
        self.classifier = mock.Mock()
        self.classifier.predict_proba.return_value = numpy.vstack((numpy.arange(0, 10)/10.0, numpy.arange(10, 0, -1)/10.0)).T

    def test_uncertain(self):
        index = dedupe.training.findUncertainPairs(None, self.classifier)
        assert index == 5

    def test_uncertain_bias(self):
        index = dedupe.training.findUncertainPairs(None, self.classifier, 1)
        print(index)
        assert index == 9
        index = dedupe.training.findUncertainPairs(None, self.classifier, 0)
        assert index == 0        


if __name__ == "__main__":
    unittest.main()
