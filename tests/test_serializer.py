import dedupe
import unittest
try:
    import json
except ImportError: 
    import simplejson as json
import StringIO
from dedupe.backport import OrderedDict

class SerializerTest(unittest.TestCase) :
  def test_writeTraining(self) :
      string = StringIO.StringIO()
      training_pairs = OrderedDict({"distinct":[(dedupe.core.frozendict({u'bar' : frozenset([u'bar']), u'foo' : u'baz'}), 
                                                 dedupe.core.frozendict({u'foo' : u'baz'}))], "match" : []})
      
      json.dump(training_pairs, 
                string, 
                default=dedupe.serializer._to_json,
                ensure_ascii = False)

      string.seek(0)

      loaded_training_pairs = json.load(string, 
                                        cls=dedupe.serializer.dedupe_decoder)

      assert loaded_training_pairs["distinct"][0] ==\
          training_pairs["distinct"][0]

      assert isinstance(loaded_training_pairs["distinct"][0][0]["bar"], 
                        frozenset)

      deduper = dedupe.Dedupe({'foo' : {'type' : 'String'}})

      string.seek(0)

      deduper._importTraining(string)
      assert repr(deduper.training_pairs) == repr(training_pairs)

      string.close()


if __name__ == "__main__":
    unittest.main()

