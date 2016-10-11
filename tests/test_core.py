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
            sample = dedupe.core.randomPairs(10**40, 10)
            assert len(w) == 2
            assert str(w[0].message) == "There may be duplicates in the sample"
            assert "Asked to sample pairs from" in str(w[1].message)
            set(sample)

        random.seed(123)
        numpy.random.seed(123)
        assert numpy.array_equal(dedupe.core.randomPairs(10**3, 1),
                                 numpy.array([(292, 413)]))



    def test_random_pair_match(self) :
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 1, 0, 10)
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 0, 0, 10)
        self.assertRaises(ValueError, dedupe.core.randomPairsMatch, 0, 1, 10)

        assert len(dedupe.core.randomPairsMatch(100, 100, 100)) == 100
        assert len(dedupe.core.randomPairsMatch(10, 10, 99)) == 99


        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pairs = dedupe.core.randomPairsMatch(10, 10, 200)
            assert str(w[0].message) == "Requested sample of size 200, only returning 100 possible pairs"

        assert len(pairs) == 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pairs = dedupe.core.randomPairsMatch(10, 10, 200)
            assert str(w[0].message) == "Requested sample of size 200, only returning 100 possible pairs"


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

    deduper = dedupe.Dedupe([{'field' : "name", 'type' : 'String'}])
    self.data_model = deduper.data_model
    self.classifier = deduper.classifier 
    self.classifier.weights = [-1.0302742719650269]
    self.classifier.bias = 4.76

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
                                         self.classifier,
                                         2)

    numpy.testing.assert_equal(scores['pairs'], 
                               self.desired_scored_pairs['pairs'])
    
    numpy.testing.assert_allclose(scores['score'], 
                                  self.desired_scored_pairs['score'], 2)



class FieldDistances(unittest.TestCase):

  def test_exact_comparator(self) :
    deduper = dedupe.Dedupe([{'field' : 'name',
                               'type' : 'Exact'}
                         ])

    record_pairs = (({'name' : 'Shmoo'}, {'name' : 'Shmee'}),
                    ({'name' : 'Shmoo'}, {'name' : 'Shmoo'}))

    numpy.testing.assert_array_almost_equal(deduper.data_model.distances(record_pairs),
                                            numpy.array([[0.0],
                                                         [1.0]]),
                                            3)

  def test_comparator(self) :
    deduper = dedupe.Dedupe([{'field' : 'type', 
                              'type' : 'Categorical',

                              'categories' : ['a', 'b', 'c']}])

    record_pairs = (({'type' : 'a'},
                     {'type' : 'b'}),
                    ({'type' : 'a'},
                     {'type' : 'c'}))

    numpy.testing.assert_array_almost_equal(deduper.data_model.distances(record_pairs),
                                            numpy.array([[ 0, 0, 1, 0, 0],
                                                         [ 0, 0, 0, 1, 0]]),
                                            3)

  def test_comparator_interaction(self) :
    deduper = dedupe.Dedupe([{'field' : 'type', 
                              'variable name' : 'type',
                              'type' : 'Categorical',
                              'categories' : ['a', 'b']},\
                             {'type' : 'Interaction',
                              'interaction variables' : ['type', 'name']},
                             {'field' : 'name',
                              'variable name' : 'name',
                              'type' : 'Exact'}])

    record_pairs = (({'name' : 'steven', 'type' : 'a'},
                     {'name' : 'steven', 'type' : 'b'}),
                    ({'name' : 'steven', 'type' : 'b'},
                     {'name' : 'steven', 'type' : 'b'}))

    numpy.testing.assert_array_almost_equal(deduper.data_model.distances(record_pairs),
                                            numpy.array([[0, 1, 1, 0, 1],
                                                         [1, 0, 1, 1, 0]]), 3)
 

if __name__ == "__main__":
    unittest.main()
