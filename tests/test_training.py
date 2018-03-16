import dedupe
import dedupe.training as training
import unittest


class TrainingTest(unittest.TestCase):
    def setUp(self):

        field_definition = [{'field': 'name', 'type': 'String'}]
        self.data_model = dedupe.Dedupe(field_definition).data_model
        self.training_pairs = {
            'match': [({"name": "Bob", "age": "50"},
                       {"name": "Bob", "age": "75"}),
                      ({"name": "Meredith", "age": "40"},
                       {"name": "Sue", "age": "10"})],
            'distinct': [({"name": "Jimmy", "age": "20"},
                          {"name": "Jimbo", "age": "21"}),
                         ({"name": "Willy", "age": "35"},
                          {"name": "William", "age": "35"}),
                         ({"name": "William", "age": "36"},
                          {"name": "William", "age": "35"})]
        }

        self.training = self.training_pairs['match'] + \
            self.training_pairs['distinct']
        self.training_records = []
        for pair in self.training:
            for record in pair:
                if record not in self.training_records:
                    self.training_records.append(record)

        self.simple = lambda x: set([str(k) for k in x
                                     if "CompoundPredicate" not in str(k)])

    def test_dedupe_coverage(self):
        predicates = self.data_model.predicates()
        blocker = dedupe.blocking.Blocker(predicates)
        blocker.indexAll({i: x for i, x in enumerate(self.training_records)})
        coverage = training.coveredPairs(blocker.predicates,
                                         self.training)
        assert self.simple(coverage.keys()).issuperset(
            set(["SimplePredicate: (tokenFieldPredicate, name)",
                 "SimplePredicate: (commonSixGram, name)",
                 "TfidfTextCanopyPredicate: (0.4, name)",
                 "SimplePredicate: (sortedAcronym, name)",
                 "SimplePredicate: (sameThreeCharStartPredicate, name)",
                 "TfidfTextCanopyPredicate: (0.2, name)",
                 "SimplePredicate: (sameFiveCharStartPredicate, name)",
                 "TfidfTextCanopyPredicate: (0.6, name)",
                 "SimplePredicate: (wholeFieldPredicate, name)",
                 "TfidfTextCanopyPredicate: (0.8, name)",
                 "SimplePredicate: (commonFourGram, name)",
                 "SimplePredicate: (firstTokenPredicate, name)",
                 "SimplePredicate: (sameSevenCharStartPredicate, name)"]))

    def test_unique(self):
        target = ([{1: 1, 2: 2}, {3: 3, 4: 4}],
                  [{3: 3, 4: 4}, {1: 1, 2: 2}])

        assert training.unique(
            [{1: 1, 2: 2}, {3: 3, 4: 4}, {1: 1, 2: 2}]) in target

    def test_remaining_cover(self):
        before = {1: {1, 2, 3}, 2: {1, 2}, 3: {3}}
        after = {1: {1, 2}, 2: {1, 2}}

        before_copy = before.copy()
        assert training.remaining_cover(before) == before
        assert training.remaining_cover(before_copy, {3}) == after
        assert before == before_copy

    def test_compound(self):
        singletons = {1: {1, 2, 3}, 2: {1, 2}, 3: {2}, 4: {5}}

        compounded = training.compound(singletons, 2)
        result = singletons.copy()
        result.update({(1, 2): {1, 2},
                       (1, 3): {2},
                       (2, 3): {2}})
        assert compounded == result

        compounded = training.compound(singletons, 3)
        result = singletons.copy()
        result.update({(1, 2): {1, 2},
                       (1, 3): {2},
                       (2, 3): {2},
                       (1, 2, 3): {2}})
        assert compounded == result

    def test_covered_pairs(self):
        p1 = lambda x, target=None: (1,)  # noqa: E 731

        cover = training.coveredPairs((p1,), [('a', 'b')] * 2)

        assert cover[p1] == {0, 1}


if __name__ == "__main__":
    unittest.main()
