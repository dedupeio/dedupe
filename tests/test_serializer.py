import codecs
import json
import sys
import unittest

import dedupe


class SerializerTest(unittest.TestCase):
    def test_writeTraining(self):
        if sys.version < "3":
            from StringIO import StringIO

            output = StringIO()
            encoded_file = codecs.EncodedFile(
                output, data_encoding="utf8", file_encoding="ascii"
            )
        else:
            from io import StringIO

            encoded_file = StringIO()

        training_pairs = {
            "distinct": [
                [
                    {
                        "bar": frozenset(["barë"]),
                        "baz": (1, 2),
                        "bang": (1, 2),
                        "foo": "baz",
                    },
                    {"foo": "baz"},
                ]
            ],
            "match": [],
        }

        json.dump(training_pairs, encoded_file, cls=dedupe.serializer.TupleEncoder)

        encoded_file.seek(0)

        loaded_training_pairs = json.load(
            encoded_file, object_hook=dedupe.serializer._from_json
        )

        assert loaded_training_pairs["distinct"][0][0] == dict(
            training_pairs["distinct"][0][0]
        )

        assert isinstance(loaded_training_pairs["distinct"][0][0]["bar"], frozenset)
        assert isinstance(loaded_training_pairs["distinct"][0][0]["baz"], tuple)

        deduper = dedupe.Dedupe([dedupe.variables.String("foo")])
        deduper.classifier.cv = False

        encoded_file.seek(0)

        deduper._read_training(encoded_file)
        print(deduper.training_pairs)
        print(training_pairs)
        assert deduper.training_pairs == training_pairs

        encoded_file.close()


if __name__ == "__main__":
    unittest.main()
