import dedupe
import unittest
try:
    import json
except ImportError: 
    import simplejson as json
import StringIO

class SerializerTest(unittest.TestCase) :
  def test_writeTraining(self) :
      string = StringIO.StringIO()
      training_pairs = {"distinct":[(dedupe.core.frozendict({'foo' : frozenset(['bar'])}), 
                                     {'foo' : 'baz'})], "match" : []}
      
      json.dump(training_pairs, 
                string, 
                default=dedupe.serializer._to_json)

      string.seek(0)

      loaded_training_pairs = json.load(string, 
                                        cls=dedupe.serializer.dedupe_decoder)

      assert loaded_training_pairs["distinct"][0] ==\
          training_pairs["distinct"][0]

      assert isinstance(loaded_training_pairs["distinct"][0][0]["foo"], 
                        frozenset)


if __name__ == "__main__":
    unittest.main()

