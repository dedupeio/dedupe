# -*- coding: utf-8 -*-
from __future__ import print_function

import dedupe
import unittest
import codecs
import json

import sys


class SerializerTest(unittest.TestCase):
    def test_writeTraining(self):
        if sys.version < '3':
            from StringIO import StringIO
            output = StringIO()
            encoded_file = codecs.EncodedFile(output,
                                              data_encoding='utf8',
                                              file_encoding='ascii')
        else:
            from io import StringIO
            encoded_file = StringIO()

        training_pairs = {u"distinct": [[{u'bar': frozenset([u'barë']),
                                          'baz': [1, 2],
                                          'bang': [1, 2],
                                          u'foo': u'baz'},
                                         {u'foo': u'baz'}]],
                          u"match": []}

        json.dump(training_pairs,
                  encoded_file,
                  default=dedupe.serializer._to_json,
                  ensure_ascii=True)

        encoded_file.seek(0)

        loaded_training_pairs = json.load(encoded_file,
                                          cls=dedupe.serializer.dedupe_decoder)

        assert loaded_training_pairs["distinct"][0][0] ==\
            dict(training_pairs["distinct"][0][0])

        assert isinstance(loaded_training_pairs["distinct"][0][0]["bar"],
                          frozenset)

        deduper = dedupe.Dedupe([{'field': 'foo', 'type': 'String'}])
        deduper.classifier.cv = False

        encoded_file.seek(0)

        deduper._read_training(encoded_file)
        print(deduper.training_pairs)
        print(training_pairs)
        assert deduper.training_pairs == training_pairs

        encoded_file.close()


if __name__ == "__main__":
    unittest.main()
