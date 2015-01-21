# -*- coding: utf-8 -*-
import dedupe
import unittest
import codecs
try:
    import json
except ImportError: 
    import simplejson as json
import StringIO
from dedupe.backport import OrderedDict

class SerializerTest(unittest.TestCase) :
  def test_writeTraining(self) :
      output = StringIO.StringIO()
      encoded_file = codecs.EncodedFile(output, data_encoding='utf8', file_encoding='ascii')

      training_pairs = OrderedDict({"distinct":[(dedupe.core.frozendict({u'bar' : frozenset([u'barë']), u'foo' : u'baz'}), 
                                                 dedupe.core.frozendict({u'foo' : u'baz'}))], "match" : []})
      
      json.dump(training_pairs, 
                encoded_file, 
                default=dedupe.serializer._to_json,
                ensure_ascii = True)

      encoded_file.seek(0)

      loaded_training_pairs = json.load(encoded_file, 
                                        cls=dedupe.serializer.dedupe_decoder)

      assert loaded_training_pairs["distinct"][0] ==\
          training_pairs["distinct"][0]

      assert isinstance(loaded_training_pairs["distinct"][0][0]["bar"], 
                        frozenset)

      deduper = dedupe.Dedupe([{'field' : 'foo', 'type' : 'String'}])

      encoded_file.seek(0)

      deduper.readTraining(output)
      assert repr(deduper.training_pairs) == repr(training_pairs)

      encoded_file.close()


if __name__ == "__main__":
    unittest.main()

