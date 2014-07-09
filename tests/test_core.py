import unittest
import dedupe
import numpy
import random
import warnings

class RandomPairsTest(unittest.TestCase) :
    def test_random_pair(self) :
        self.assertRaises(ValueError, dedupe.core.randomPairs, 1, 10)
        assert dedupe.core.randomPairs(10, 10).any()
        random.seed(123)
        numpy.random.seed(123)
        random_pairs = dedupe.core.randomPairs(10, 5)
        assert numpy.array_equal(random_pairs, 
                                 numpy.array([[ 0,  3],
                                              [ 3,  8],
                                              [ 4,  9],
                                              [ 5,  9],
                                              [ 2,  3]]))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dedupe.core.randomPairs(10, 10**6)
            assert len(w) == 1
            assert str(w[-1].message) == "Requested sample of size 1000000, only returning 45 possible pairs"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dedupe.core.randomPairs(10**40, 10)
            assert len(w) == 2
            assert str(w[0].message) == "There may be duplicates in the sample"
            assert "Asked to sample pairs from" in str(w[1].message)

        random.seed(123)
        numpy.random.seed(123)
        assert numpy.array_equal(dedupe.core.randomPairs(11**9, 1),
                                 numpy.array([[1228959102, 1840268610]]))



    def test_random_pair_match(self) :
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 1, 0, 10)
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 0, 0, 10)
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 0, 1, 10)

        assert len(dedupe.core.randomPairsMatch(100, 100, 100)) == 100
        assert len(dedupe.core.randomPairsMatch(10, 10, 99)) == 99


        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pairs = dedupe.core.randomPairsMatch(10, 10, 200)
            assert len(w) == 1
            assert str(w[-1].message) == "Requested sample of size 200, only returning 100 possible pairs"

        assert len(pairs) == 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pairs = dedupe.core.randomPairsMatch(10, 10, 200)
            assert len(w) == 1
            assert str(w[-1].message) == "Requested sample of size 200, only returning 100 possible pairs"


        random.seed(123)
        numpy.random.seed(123)
        pairs = dedupe.core.randomPairsMatch(10, 10, 10)
        assert pairs == set([(7, 3), (3, 3), (2, 9), (6, 0), (2, 0), 
                             (1, 9), (9, 4), (0, 4), (1, 0), (1, 1)])


class ScoreDuplicates(unittest.TestCase):
  def setUp(self) :
    random.seed(123)
    empty_set = set([])

    self.records = iter([(('1', {'name': 'Margret', 'age': '32'}, empty_set), 
                          ('2', {'name': 'Marga', 'age': '33'}, empty_set)), 
                         (('2', {'name': 'Marga', 'age': '33'}, empty_set), 
                          ('3', {'name': 'Maria', 'age': '19'}, empty_set)), 
                         (('4', {'name': 'Maria', 'age': '19'}, empty_set), 
                          ('5', {'name': 'Monica', 'age': '39'}, empty_set)), 
                         (('6', {'name': 'Monica', 'age': '39'}, empty_set), 
                          ('7', {'name': 'Mira', 'age': '47'}, empty_set)),
                         (('8', {'name': 'Mira', 'age': '47'}, empty_set), 
                          ('9', {'name': 'Mona', 'age': '9'}, empty_set)),
                        ])

    self.data_model = dedupe.Dedupe({"name" : {'type' : 'String'}}, ()).data_model
    self.data_model['fields']['name'].weight = -1.0302742719650269
    self.data_model['bias'] = 4.76

    score_dtype = [('pairs', 'S4', 2), ('score', 'f4', 1)]

    self.desired_scored_pairs = numpy.array([(('1', '2'), 0.96), 
                                             (['2', '3'], 0.96), 
                                             (['4', '5'], 0.78), 
                                             (['6', '7'], 0.72), 
                                             (['8', '9'], 0.84)], 
                                            dtype=score_dtype)



  def test_score_duplicates(self):
    scores = dedupe.core.scoreDuplicates(self.records,
                                         self.data_model,
                                         2)

    numpy.testing.assert_equal(scores['pairs'], 
                               self.desired_scored_pairs['pairs'])
    
    numpy.testing.assert_allclose(scores['score'], 
                                  self.desired_scored_pairs['score'], 2)



class FieldDistances(unittest.TestCase):
  def test_field_distance_simple(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe({'name' : {'type' :'String'},
                             'source' : {'type' : 'Source',
                                         'Source Names' : ['a', 'b']}}, [])

    record_pairs = (({'name' : 'steve', 'source' : 'a'}, 
                     {'name' : 'steven', 'source' : 'a'}),)

    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[0, 0.647, 0, 0, 0]]), 3)

    record_pairs = (({'name' : 'steve', 'source' : 'b'}, 
                     {'name' : 'steven', 'source' : 'b'}),)
    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[1, 0.647, 0, 0.647, 0]]), 3)

    record_pairs = (({'name' : 'steve', 'source' : 'a'}, 
                     {'name' : 'steven', 'source' : 'b'}),)
    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[0, 0.647, 1, 0, 0.647]]), 3)

  def test_comparator(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe({'type' : {'type' : 'Categorical',
                                       'Categories' : ['a', 'b', 'c']}
                             }, [])

    record_pairs = (({'type' : 'a'},
                     {'type' : 'b'}),
                    ({'type' : 'a'},
                     {'type' : 'c'}))

    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[ 0, 0, 1, 0, 0],
                                                         [ 0, 0, 0, 1, 0]]),
                                            3)

    deduper = dedupe.Dedupe({'type' : {'type' : 'Categorical',
                                       'Categories' : ['a', 'b', 'c']},
                             'source' : {'type' : 'Source',
                                         'Source Names' : ['foo', 'bar']}
                             }, [])

    record_pairs = (({'type' : 'a',
                      'source' : 'bar'},
                     {'type' : 'b',
                      'source' : 'bar'}),
                    ({'type' : 'a', 
                      'source' : 'foo'},
                     {'type' : 'c',
                      'source' : 'bar'}))


    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
         numpy.array([[ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.],
                      [ 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.]]),
                                            3)

 

  def test_field_distance_interaction(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe({'first_name' : {'type' :'String'},
                             'last_name' : {'type' : 'String'},
                             'first-last' : {'type' : 'Interaction', 
                                             'Interaction Fields' : ['first_name', 
                                                                     'last_name']},
                             'source' : {'type' : 'Source',
                                         'Source Names' : ['a', 'b']}
                           }, [])

    record_pairs = (({'first_name' : 'steve', 
                      'last_name' : 'smith', 
                      'source' : 'b'}, 
                     {'first_name' : 'steven', 
                      'last_name' : 'smith', 
                      'source' : 'b'}),)

    # ['source', 'first_name', 'last_name', 'different sources',
    # 'first-last', 'source:first_name', 'different sources:first_name',
    # 'source:last_name', 'different sources:last_name',
    # 'source:first-last', 'different sources:first-last']
    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[ 1.0,  
                                                           0.647,  
                                                           0.5,  
                                                           0.0,
                                                           0.323,
                                                           0.647,
                                                           0.0,
                                                           0.5,
                                                           0.0,
                                                           0.323,
                                                           0.0]]),
                                            3)
if __name__ == "__main__":
    unittest.main()
