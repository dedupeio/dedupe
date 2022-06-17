import random
import unittest

import pytest

import dedupe

SAMPLE = [
    ({"name": "Bob", "age": "50"}, {"name": "Charlie", "age": "75"}),
    ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
    ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
    ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
]


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.data_model = dedupe.datamodel.DataModel(
            [{"field": "name", "type": "String"}, {"field": "age", "type": "String"}]
        )

    def test_AL(self):
        random.seed(1111111111110)
        original_N = len(SAMPLE)
        active_learner = dedupe.labeler.MatchLearner(self.data_model)
        active_learner.candidates = SAMPLE
        assert len(active_learner) == original_N

        active_learner.pop()
        assert len(active_learner) == original_N - 1

        active_learner.pop()
        assert len(active_learner) == original_N - 2

        active_learner.pop()
        assert len(active_learner) == original_N - 3

        active_learner.pop()

        with pytest.raises(IndexError):
            active_learner.pop()


if __name__ == "__main__":
    unittest.main()
