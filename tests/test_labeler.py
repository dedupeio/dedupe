import random
import unittest

import pytest

from dedupe import datamodel
from dedupe import labeler

SAMPLE = {
    1: {"name": "Meredith", "age": "40"},
    2: {"name": "Sue", "age": "10"},
    3: {"name": "Willy", "age": "35"},
    4: {"name": "William", "age": "35"},
    5: {"name": "Jimmy", "age": "20"},
    6: {"name": "Jimbo", "age": "21"},
}


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.data_model = datamodel.DataModel(
            [{"field": "name", "type": "String"}, {"field": "age", "type": "String"}]
        )

    def test_AL(self):
        random.seed(1111111111110)
        # Even with random seed, the order of the following seem to be random,
        # so we shouldn't test for exact order.
        EXPECTED_CANDIDATES = [
            ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
            ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
            ({"name": "William", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "William", "age": "35"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "Sue", "age": "10"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Jimmy", "age": "20"}),
        ]
        active_learner = labeler.DedupeDisagreementLearner(self.data_model, SAMPLE, [])
        for i in range(len(EXPECTED_CANDIDATES), 0, -1):
            assert len(active_learner) == i
            active_learner.pop()
        with pytest.raises(IndexError):
            active_learner.pop()


if __name__ == "__main__":
    unittest.main()
