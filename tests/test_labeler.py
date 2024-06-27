import random
import unittest

import pytest

import dedupe
from dedupe import datamodel, labeler
from dedupe._typing import RecordDictPair

SAMPLE = {
    1: {"name": "Meredith", "age": "40"},
    2: {"name": "Sue", "age": "10"},
    3: {"name": "Willy", "age": "35"},
    4: {"name": "William", "age": "35"},
    5: {"name": "Jimmy", "age": "20"},
    6: {"name": "Jimbo", "age": "21"},
}


def freeze_record_pair(record_pair: RecordDictPair):
    rec1, rec2 = record_pair
    return (frozenset(rec1.items()), frozenset(rec2.items()))


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.data_model = datamodel.DataModel(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )

    def test_AL(self):
        random.seed(1111111111110)
        # Even with random seed, the order of the following seem to be random,
        # so we shouldn't test for exact order.
        EXPECTED_CANDIDATES = [
            ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
            ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "William", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Sue", "age": "10"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "William", "age": "35"}, {"name": "Jimbo", "age": "21"}),
        ]
        EXPECTED_CANDIDATES = {freeze_record_pair(pair) for pair in EXPECTED_CANDIDATES}
        active_learner = labeler.DedupeDisagreementLearner(
            self.data_model.predicates, self.data_model.distances, SAMPLE, []
        )
        actual_candidates = set()
        for i in range(len(EXPECTED_CANDIDATES), 0, -1):
            assert len(active_learner) == i
            record_pair = freeze_record_pair(active_learner.pop())
            actual_candidates.add(record_pair)
        assert actual_candidates == EXPECTED_CANDIDATES
        with pytest.raises(IndexError):
            active_learner.pop()


if __name__ == "__main__":
    unittest.main()
