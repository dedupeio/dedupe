import dedupe
import unittest
try:
    import unittest.mock as mock
except ImportError:
    import mock
import numpy

SAMPLE = [({"name": "Bob", "age": "50"}, {"name": "Charlie", "age": "75"}),
          ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}), 
          ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
          ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"})]


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.data_model = dedupe.datamodel.DataModel([{'field' : 'name',
                                                       'type'  : 'String'},
                                                      {'field' : 'age',
                                                       'type'  : 'String'}])
        self.classifier = mock.Mock()
        self.classifier.predict_proba = lambda x : numpy.vstack((numpy.arange(0, len(x))/float(len(x)), numpy.arange(len(x), 0, -1)/float(len(x)))).T

    def test_AL(self):
        original_N = len(SAMPLE)
        active_learner = dedupe.training.ActiveLearning(SAMPLE,
                                                        self.data_model, 1)
        assert len(active_learner) == original_N
        pair = active_learner.uncertainPairs(self.classifier, 0.5)
        print(self.classifier.predict_proba(xrange(10)))
        assert pair == [({"name": "Jimmy", "age": "20"},
                         {"name": "Jimbo", "age": "21"})]
        assert len(active_learner) == original_N - 1

        pair = active_learner.uncertainPairs(self.classifier, 1)
        assert pair == [({"name": "Willy", "age": "35"},
                         {"name": "William", "age": "35"})]
        assert len(active_learner) == original_N - 2

        pair = active_learner.uncertainPairs(self.classifier, 0)
        assert pair == [({"name": "Bob", "age": "50"},
                         {"name": "Charlie", "age": "75"})]
        assert len(active_learner) == original_N - 3


        

if __name__ == "__main__":
    unittest.main()
