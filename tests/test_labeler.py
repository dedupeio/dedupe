import dedupe
import unittest
import random
import pytest

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
        active_learner = dedupe.labeler.RLRLearner(self.data_model)
        active_learner.candidates = SAMPLE
        assert len(active_learner) == original_N
        pair = active_learner.pop()
        print(pair)
        assert pair == (
            {"name": "Willy", "age": "35"},
            {"name": "William", "age": "35"},
        )

        assert len(active_learner) == original_N - 1

        pair = active_learner.pop()
        print(pair)
        assert pair == ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"})
        assert len(active_learner) == original_N - 2

        pair = active_learner.pop()
        assert pair == ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"})

        assert len(active_learner) == original_N - 3

        active_learner.pop()

        with pytest.raises(IndexError):
            active_learner.pop()


if __name__ == "__main__":
    unittest.main()
