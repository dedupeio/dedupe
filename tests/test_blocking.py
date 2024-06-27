import unittest
from collections import defaultdict

import dedupe


class BlockingTest(unittest.TestCase):
    def setUp(self):
        field_definition = [{"field": "name", "type": "String"}]
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


class TfidfTest(unittest.TestCase):
    def setUp(self):
        self.data_d = {
            100: {"name": "Bob", "age": "50", "dataset": 0},
            105: {"name": "Charlie", "age": "75", "dataset": 1},
            110: {"name": "Meredith", "age": "40", "dataset": 1},
            115: {"name": "Sue", "age": "10", "dataset": 0},
            120: {"name": "Jimbo", "age": "21", "dataset": 0},
            125: {"name": "Jimbo", "age": "21", "dataset": 0},
            130: {"name": "Willy", "age": "35", "dataset": 0},
            135: {"name": "Willy", "age": "35", "dataset": 1},
            140: {"name": "Martha", "age": "19", "dataset": 1},
            145: {"name": "Kyle", "age": "27", "dataset": 0},
        }

    def test_unconstrained_inverted_index(self):
        blocker = dedupe.blocking.Fingerprinter(
            [dedupe.predicates.TfidfTextSearchPredicate(0.0, "name")]
        )

        blocker.index({record["name"] for record in self.data_d.values()}, "name")

        blocks = defaultdict(set)

        for block_key, record_id in blocker(self.data_d.items()):
            blocks[block_key].add(record_id)

        blocks = {frozenset(block) for block in blocks.values() if len(block) > 1}

        assert blocks == {frozenset([120, 125]), frozenset([130, 135])}


class TfIndexUnindex(unittest.TestCase):
    def setUp(self):
        data_d = {
            100: {"name": "Bob", "age": "50", "dataset": 0},
            105: {"name": "Charlie", "age": "75", "dataset": 1},
            110: {"name": "Meredith", "age": "40", "dataset": 1},
            115: {"name": "Sue", "age": "10", "dataset": 0},
            120: {"name": "Jimbo", "age": "21", "dataset": 0},
            125: {"name": "Jimbo", "age": "21", "dataset": 0},
            130: {"name": "Willy", "age": "35", "dataset": 0},
            135: {"name": "Willy", "age": "35", "dataset": 1},
            140: {"name": "Martha", "age": "19", "dataset": 1},
            145: {"name": "Kyle", "age": "27", "dataset": 0},
        }

        self.blocker = dedupe.blocking.Fingerprinter(
            [dedupe.predicates.TfidfTextSearchPredicate(0.0, "name")]
        )

        self.records_1 = {
            record_id: record
            for record_id, record in data_d.items()
            if record["dataset"] == 0
        }

        self.fields_2 = {
            record_id: record["name"]
            for record_id, record in data_d.items()
            if record["dataset"] == 1
        }

    def test_index(self):
        self.blocker.index(set(self.fields_2.values()), "name")

        blocks = defaultdict(set)

        for block_key, record_id in self.blocker(self.records_1.items()):
            blocks[block_key].add(record_id)

        assert list(blocks.items())[0][1] == {130}

    def test_doubled_index(self):
        self.blocker.index(self.fields_2.values(), "name")
        self.blocker.index(self.fields_2.values(), "name")

        blocks = defaultdict(set)

        for block_key, record_id in self.blocker(self.records_1.items()):
            blocks[block_key].add(record_id)

        result = list(blocks.items())

        assert len(result) == 1

        assert result[0][1] == {130}

    def test_unindex(self):
        self.blocker.index(self.fields_2.values(), "name")
        self.blocker.unindex(self.fields_2.values(), "name")

        blocks = defaultdict(set)

        for block_key, record_id in self.blocker(self.records_1.items()):
            blocks[block_key].add(record_id)

        assert len(blocks.values()) == 0


if __name__ == "__main__":
    unittest.main()
