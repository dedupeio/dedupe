import unittest

import dedupe
import dedupe.branch_and_bound as branch_and_bound
import dedupe.training as training


class TrainingTest(unittest.TestCase):
    def setUp(self):
        field_definition = [dedupe.variables.String("name")]
        self.data_model = dedupe.Dedupe(field_definition).data_model
        self.training_pairs = {
            "match": [
                ({"name": "Bob", "age": "50"}, {"name": "Bob", "age": "75"}),
                ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
            ],
            "distinct": [
                ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
                ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
                ({"name": "William", "age": "36"}, {"name": "William", "age": "35"}),
            ],
        }

        self.training = self.training_pairs["match"] + self.training_pairs["distinct"]
        self.training_records = []
        for pair in self.training:
            for record in pair:
                if record not in self.training_records:
                    self.training_records.append(record)

        self.simple = lambda x: {str(k) for k in x if "CompoundPredicate" not in str(k)}

        self.block_learner = training.BlockLearner
        self.block_learner.blocker = dedupe.blocking.Fingerprinter(
            self.data_model.predicates
        )
        self.block_learner.blocker.index_all(
            {i: x for i, x in enumerate(self.training_records)}
        )

    def test_dedupe_coverage(self):
        coverage = self.block_learner.cover(self.block_learner, self.training)
        assert self.simple(coverage.keys()).issuperset(
            {
                "SimplePredicate: (tokenFieldPredicate, name)",
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
                "SimplePredicate: (sameSevenCharStartPredicate, name)",
            }
        )

    def test_uncovered_by(self):
        before = {1: frozenset({1, 2, 3}), 2: frozenset({1, 2}), 3: frozenset({3})}
        after = {1: frozenset({1, 2}), 2: frozenset({1, 2})}

        before_copy = before.copy()

        assert branch_and_bound._uncovered_by(before, frozenset()) == before
        assert branch_and_bound._uncovered_by(before, frozenset({3})) == after
        assert before == before_copy

    def test_covered_pairs(self):
        p1 = lambda x, target=None: frozenset((1,))  # noqa: E 731

        self.block_learner.blocker.predicates = (p1,)
        cover = self.block_learner.cover(self.block_learner, [("a", "b")] * 2)

        assert cover[p1] == {0, 1}


if __name__ == "__main__":
    unittest.main()
