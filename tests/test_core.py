import unittest
import dedupe
import numpy
import random
import warnings


class RandomPairsTest(unittest.TestCase) :
    def test_random_pair(self) :
        self.assertRaises(ValueError, dedupe.core.randomPairs, 1, 10)
        assert dedupe.core.randomPairs(10, 10)
        random.seed(123)
        numpy.random.seed(123)
        random_pairs = dedupe.core.randomPairs(10, 5)
        assert random_pairs == [( 0,  3),
                                ( 3,  8),
                                ( 4,  9),
                                ( 5,  9),
                                ( 2,  3)]

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

    long_string ='asa;sasdfjasdio;fio;asdnfasdvnvao;asduifvnavjasdfasdfasfasasdfasdfasdfasdfasdfsdfasgnuavpidcvaspdivnaspdivninasduinguipghauipsdfnvaspfighapsdifnasdifnasdpighuignpaguinpgiasidfjasdfjsdofgiongag'

    self.records = iter([((long_string, {'name': 'Margret', 'age': '32'}, empty_set), 
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

    self.data_model = dedupe.Dedupe([{'field' : "name", 'type' : 'String'}], ()).data_model
    self.data_model['fields'][0].weight = -1.0302742719650269
    self.data_model['bias'] = 4.76

    score_dtype = [('pairs', '<U192', 2), ('score', 'f4', 1)]

    self.desired_scored_pairs = numpy.array([((long_string, '2'), 0.96), 
                                             (['2', '3'], 0.96), 
                                             (['4', '5'], 0.78), 
                                             (['6', '7'], 0.72), 
                                             (['8', '9'], 0.84)], 
                                            dtype=score_dtype)



  def test_score_duplicates(self):
    scores = dedupe.core.scoreDuplicates(self.records,
                                         self.data_model,
                                         2)

    print scores.dtype


    numpy.testing.assert_equal(scores['pairs'], 
                               self.desired_scored_pairs['pairs'])
    
    numpy.testing.assert_allclose(scores['score'], 
                                  self.desired_scored_pairs['score'], 2)



class FieldDistances(unittest.TestCase):
  def test_field_distance_simple(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe([{'field' : 'name' , 'type' :'String'},
                             {'field' : 'source', 'type' : 'Source',
                              'sources' : ['a', 'b']}], [])

    record_pairs = (({'name' : 'steve', 'source' : 'a'}, 
                     {'name' : 'steven', 'source' : 'a'}),)


    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[0, 0.647, 0, 0, 0]]), 3)

    record_pairs = (({'name' : 'steve', 'source' : 'b'}, 
                     {'name' : 'steven', 'source' : 'b'}),)
    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[1, 0.647, 0, 0, 0.647]]), 3)

    record_pairs = (({'name' : 'steve', 'source' : 'a'}, 
                     {'name' : 'steven', 'source' : 'b'}),)
    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[0, 0.647, 1, 0.647, 0]]), 3)

  def test_exact_comparator(self) :
    fieldDistances = dedupe.core.fieldDistances      
    deduper = dedupe.Dedupe([{'field' : 'name', 
                              'type' : 'String'},
                             {'field' : 'name',
                              'type' : 'Exact'}
                         ])

    record_pairs = (({'name' : 'Shmoo'}, {'name' : 'Shmee'}),
                    ({'name' : 'Shmoo'}, {'name' : 'Shmoo'}))

    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[ 2.5, 0.0],
                                                         [ 0.5, 1.0]]),
                                            3)

  def test_comparator(self) :
    fieldDistances = dedupe.core.fieldDistances      

    deduper = dedupe.Dedupe([{'field' : 'type', 
                              'type' : 'Categorical',
                              'categories' : ['a', 'b', 'c']}]
                             , [])

    record_pairs = (({'type' : 'a'},
                     {'type' : 'b'}),
                    ({'type' : 'a'},
                     {'type' : 'c'}))

    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[ 0, 0, 1, 0, 0],
                                                         [ 0, 0, 0, 1, 0]]),
                                            3)

    deduper = dedupe.Dedupe([{'field' : 'type', 'type' : 'Categorical',
                                       'categories' : ['a', 'b', 'c']},
                             {'field' : 'source', 'type' : 'Source',
                                         'sources' : ['foo', 'bar']}
                             ], [])

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
         numpy.array([[ 0.0,  1.0,  0.0,  1.0,  
                        0.0,  0.0,  0.0,  0.0,  
                        0.0,  0.0,  1.0,  0.0,  
                        0.0,  0.0,  0.0,  0.0,  0.0],
                      [ 0.0,  0.0,  0.0,  0.0,  
                        1.0,  0.0,  1.0,  0.0,  
                        0.0,  0.0,  0.0,  0.0,  
                        0.0,  0.0,  0.0,  0.0,  0.0]]), 3)

 

  def test_field_distance_interaction(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe([{'field' : 'first_name', 
                              'variable name' : 'first_name', 
                              'type' :'String'},
                             {'field' : 'last_name', 
                              'variable name' : 'last_name', 
                              'type' : 'String'},
                             {'type' : 'Interaction', 
                              'interaction variables' : ['first_name', 
                                                      'last_name']},
                             {'field' : 'source',
                              'type' : 'Source',
                              'sources' : ['a', 'b']}
                         ], [])

    record_pairs = (({'first_name' : 'steve', 
                      'last_name' : 'smith', 
                      'source' : 'b'}, 
                     {'first_name' : 'steven', 
                      'last_name' : 'smith', 
                      'source' : 'b'}),)


    numpy.testing.assert_array_almost_equal(fieldDistances(record_pairs, 
                                                           deduper.data_model),
                                            numpy.array([[ 0.647,
                                                           0.5,
                                                           1.0,
                                                           0.0,
                                                           0.323,
                                                           0.647,
                                                           0.0,
                                                           0.0,
                                                           0.0,
                                                           0.5,
                                                           0.323]]),
                                            3)
if __name__ == "__main__":
    unittest.main()
