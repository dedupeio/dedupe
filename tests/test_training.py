import dedupe
import unittest
import numpy
import random

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
    def test_AL(self):
        random.seed(111111111110)
        original_N = len(SAMPLE)
        active_learner = dedupe.labeler.RLRLearner(self.data_model)
        active_learner.candidates = SAMPLE
        active_learner.distances = active_learner.transform(SAMPLE)
        active_learner._init_rlr()
        assert len(active_learner) == original_N
        pair = active_learner.get()
        print(pair)
        assert pair == [({"name": "Willy", "age": "35"},
                         {"name": "William", "age": "35"})]

        assert len(active_learner) == original_N - 1

        pair = active_learner.get()
        assert pair == [({"name": "Jimmy", "age": "20"},
                         {"name": "Jimbo", "age": "21"})]
        assert len(active_learner) == original_N - 2

        pair = active_learner.get()
        assert pair == [({"name": "Meredith", "age": "40"},
                         {"name": "Sue", "age": "10"})] 

        assert len(active_learner) == original_N - 3


        

if __name__ == "__main__":
    unittest.main()
