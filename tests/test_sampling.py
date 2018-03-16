import unittest
import dedupe
import dedupe.sampling
import dedupe.predicates
import dedupe.api
from collections import deque

data_dict = {'1': {'name': 'Bob', 'age': '51'},
             '2': {'name': 'Linda', 'age': '50'},
             '3': {'name': 'Gene', 'age': '12'},
             '4': {'name': 'Tina', 'age': '15'},
             '5': {'name': 'Bob B.', 'age': '51'},
             '6': {'name': 'bob belcher', 'age': '51'},
             '7': {'name': 'linda ', 'age': '50'}}


class DedupeSampling(unittest.TestCase):
    def setUp(self):
        field_definition = [{'field': 'name', 'type': 'String'},
                            {'field': 'age', 'type': 'String'}]
        self.deduper = dedupe.Dedupe(field_definition)

    def test_even_split(self):
        assert sum(dedupe.sampling.evenSplits(10, 10)) == 10
        assert sum(dedupe.sampling.evenSplits(10, 1)) == 10
        assert sum(dedupe.sampling.evenSplits(10, 4)) == 10

    def test_sample_predicate(self):
        items = data_dict.items()
        pred = dedupe.predicates.SimplePredicate(dedupe.predicates.sameThreeCharStartPredicate,
                                                 'name')
        assert dedupe.sampling.dedupeSamplePredicate(10,
                                                     pred,
                                                     items) == [('1', '5')]

    def test_sample_predicates(self):
        items = deque(data_dict.items())
        pred = dedupe.predicates.SimplePredicate(dedupe.predicates.sameThreeCharStartPredicate,
                                                 'name')

        assert list(dedupe.sampling.dedupeSamplePredicates(10,
                                                           [pred],
                                                           items)) == [[('1', '5')]]

    def test_blockedSample(self):
        pred = dedupe.predicates.SimplePredicate(dedupe.predicates.sameThreeCharStartPredicate,
                                                 'name')
        assert len(dedupe.sampling.dedupeBlockedSample(10,
                                                       [pred],
                                                       deque(data_dict.items()))) == 1
