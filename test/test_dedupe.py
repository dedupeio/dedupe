import dedupe
import unittest
import numpy
import random
import itertools
import warnings
import multiprocessing

class CoreTest(unittest.TestCase):
  def setUp(self) :
    random.seed(123)

    self.records = iter([(('1', {'name': 'Margret', 'age': '32'}), 
                          ('2', {'name': 'Marga', 'age': '33'})), 
                         (('2', {'name': 'Marga', 'age': '33'}), 
                          ('3', {'name': 'Maria', 'age': '19'})), 
                         (('4', {'name': 'Maria', 'age': '19'}), 
                          ('5', {'name': 'Monica', 'age': '39'})), 
                         (('6', {'name': 'Monica', 'age': '39'}), 
                          ('7', {'name': 'Mira', 'age': '47'})),
                         (('8', {'name': 'Mira', 'age': '47'}), 
                          ('9', {'name': 'Mona', 'age': '9'})),
                        ])

    self.normalizedAffineGapDistance = dedupe.affinegap.normalizedAffineGapDistance
    self.data_model = {}
    self.data_model['fields'] = dedupe.backport.OrderedDict()
    v = {}
    v.update({'Has Missing': False, 'type': 'String', 'comparator': self.normalizedAffineGapDistance, \
              'weight': -1.0302742719650269})
    self.data_model['fields']['name'] = v
    self.data_model['bias'] = 4.76

    score_dtype = [('pairs', 'S1', 2), ('score', 'f4', 1)]
    self.desired_scored_pairs = numpy.array([(['1', '2'], 0.96), (['2', '3'], 0.96), \
                                             (['4', '5'], 0.78), (['6', '7'], 0.72), \
                                             (['8', '9'], 0.84)], dtype=score_dtype)


  def test_random_pair(self) :
    self.assertRaises(ValueError, dedupe.core.randomPairs, 1, 10)
    assert dedupe.core.randomPairs(10, 10).any()
    assert dedupe.core.randomPairs(10*1000000000, 10).any()
    assert numpy.array_equal(dedupe.core.randomPairs(10, 5), 
                             numpy.array([[ 1,  8],
                                          [ 5,  7],
                                          [ 1,  2],
                                          [ 3,  7],
                                          [ 2,  9]]))

  def test_score_duplicates(self):
    actual_scored_pairs_str = dedupe.core.scoreDuplicates(self.records,
                                                          'S1',
                                                          self.data_model,
                                                          multiprocessing.Pool(processes=1))

    scores_str = numpy.around(actual_scored_pairs_str['score'], decimals=2)

    numpy.testing.assert_almost_equal(self.desired_scored_pairs['score'], scores_str)
    numpy.testing.assert_equal(self.desired_scored_pairs['pairs'], actual_scored_pairs_str['pairs'])

class ConvenienceTest(unittest.TestCase):
  def setUp(self):
    self.data_d = {  100 : {"name": "Bob", "age": "50"},
                     105 : {"name": "Charlie", "age": "75"},
                     110 : {"name": "Meredith", "age": "40"},
                     115 : {"name": "Sue", "age": "10"}, 
                     120 : {"name": "Jimmy", "age": "20"},
                     125 : {"name": "Jimbo", "age": "21"},
                     130 : {"name": "Willy", "age": "35"},
                     135 : {"name": "William", "age": "35"},
                     140 : {"name": "Martha", "age": "19"},
                     145 : {"name": "Kyle", "age": "27"},
                  }
    random.seed(123)

  def test_data_sample(self):
    assert dedupe.convenience.dataSample(self.data_d,5) == \
            (({'age': '27', 'name': 'Kyle'}, {'age': '50', 'name': 'Bob'}),
            ({'age': '27', 'name': 'Kyle'}, {'age': '35', 'name': 'William'}),
            ({'age': '10', 'name': 'Sue'}, {'age': '35', 'name': 'William'}),
            ({'age': '27', 'name': 'Kyle'}, {'age': '20', 'name': 'Jimmy'}),
            ({'age': '75', 'name': 'Charlie'}, {'age': '21', 'name': 'Jimbo'}))

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      dedupe.convenience.dataSample(self.data_d,10000)
      assert len(w) == 1
      assert str(w[-1].message) == "Requested sample of size 10000, only returning 45 possible pairs"


 
class DedupeClassTest(unittest.TestCase):
  def test_initialize(self) :
    dist = dedupe.affinegap.normalizedAffineGapDistance
    deduper = dedupe.Dedupe({'name' : {'type' : 'String'}})
    assert deduper.data_model['fields']['name'] == {'Has Missing': False, 
                                                    'type': 'String', 
                                                    'comparator': dist}

    deduper = dedupe.Dedupe({'name' : {'type' : 'String',
                                       'Has Missing' : True}})
    assert deduper.data_model['fields']['name'] == {'Has Missing': True, 
                                                    'type': 'String', 
                                                    'comparator': dist }

    deduper = dedupe.Dedupe({'name' : {'type' : 'Source',
                                       'Source Names' : ['a', 'b'],
                                       'Has Missing' : True}})

    source_comparator = deduper.data_model['fields']['name']['comparator']
    assert source_comparator('a', 'a') == 0
    assert source_comparator('b', 'b') == 1
    assert source_comparator('a', 'b') == 2
    assert source_comparator('b', 'a') == 2
    self.assertRaises(ValueError, source_comparator, 'b', 'c')
    self.assertRaises(ValueError, source_comparator, '', 'c')
    assert numpy.isnan(source_comparator('', 'b'))

  def test_base_predicates(self) :
    deduper = dedupe.Dedupe({'name' : {'type' : 'String'}})
    string_predicates = (dedupe.predicates.wholeFieldPredicate,
                         dedupe.predicates.tokenFieldPredicate,
                         dedupe.predicates.commonIntegerPredicate,
                         dedupe.predicates.sameThreeCharStartPredicate,
                         dedupe.predicates.sameFiveCharStartPredicate,
                         dedupe.predicates.sameSevenCharStartPredicate,
                         dedupe.predicates.nearIntegersPredicate,
                         dedupe.predicates.commonFourGram,
                         dedupe.predicates.commonSixGram)

    tfidf_string_predicates = tuple([dedupe.tfidf.TfidfPredicate(threshold)
                                     for threshold
                                     in [0.2, 0.4, 0.6, 0.8]])

    assert deduper.blocker_types == {'String' : string_predicates + tfidf_string_predicates}

  


class AffineGapTest(unittest.TestCase):
  def setUp(self):
    self.affineGapDistance = dedupe.affinegap.affineGapDistance
    self.normalizedAffineGapDistance = dedupe.affinegap.normalizedAffineGapDistance
    
  def test_affine_gap_correctness(self):
    assert self.affineGapDistance('a', 'b', -5, 5, 5, 1, 0.5) == 5
    assert self.affineGapDistance('ab', 'cd', -5, 5, 5, 1, 0.5) == 10
    assert self.affineGapDistance('ab', 'cde', -5, 5, 5, 1, 0.5) == 13
    assert self.affineGapDistance('a', 'cde', -5, 5, 5, 1, 0.5) == 8.5
    assert self.affineGapDistance('a', 'cd', -5, 5, 5, 1, 0.5) == 8
    assert self.affineGapDistance('b', 'a', -5, 5, 5, 1, 0.5) == 5
    assert self.affineGapDistance('a', 'a', -5, 5, 5, 1, 0.5) == -5
    assert numpy.isnan(self.affineGapDistance('a', '', -5, 5, 5, 1, 0.5))
    assert numpy.isnan(self.affineGapDistance('', '', -5, 5, 5, 1, 0.5))
    assert self.affineGapDistance('aba', 'aaa', -5, 5, 5, 1, 0.5) == -5
    assert self.affineGapDistance('aaa', 'aba', -5, 5, 5, 1, 0.5) == -5
    assert self.affineGapDistance('aaa', 'aa', -5, 5, 5, 1, 0.5) == -7
    assert self.affineGapDistance('aaa', 'a', -5, 5, 5, 1, 0.5) == -1.5
    assert numpy.isnan(self.affineGapDistance('aaa', '', -5, 5, 5, 1, 0.5))
    assert self.affineGapDistance('aaa', 'abba', -5, 5, 5, 1, 0.5) == 1
    
  def test_normalized_affine_gap_correctness(self):
    assert numpy.isnan(self.normalizedAffineGapDistance('', '', -5, 5, 5, 1, 0.5))
    
class ClusteringTest(unittest.TestCase):
  def setUp(self):
    # Fully connected star network
    self.dupes = (((1,2), .86),
                  ((1,3), .72),
                  ((1,4), .2),
                  ((1,5), .6),                 
                  ((2,3), .86),
                  ((2,4), .2),
                  ((2,5), .72),
                  ((3,4), .3),
                  ((3,5), .5),
                  ((4,5), .72))
    #Dupes with Ids as String
    self.str_dupes = ((('1', '2'), .86),
                      (('1', '3'), .72),
                      (('1', '4'), .2),
                      (('1', '5'), .6),
                      (('2', '3'), .86),
                      (('2', '4'), .2),
                      (('2', '5'), .72),
                      (('3', '4'), .3),
                      (('3', '5'), .5),
                      (('4', '5'), .72))

            
  def test_hierarchical(self):
    hierarchical = dedupe.clustering.cluster
    assert hierarchical(self.dupes, 'i4', 1) == []
    assert hierarchical(self.dupes, 'i4', 0.5) == [set([1, 2, 3]), set([4,5])]
    assert hierarchical(self.dupes, 'i4', 0) == [set([1, 2, 3, 4, 5])]
    assert hierarchical(self.str_dupes, 'S1', 1) == []
    assert hierarchical(self.str_dupes,'S1', 0.5) == [set(['1', '2', '3']), set(['4','5'])]
    assert hierarchical(self.str_dupes,'S1', 0) == [set(['1', '2', '3', '4', '5'])]

class BlockingTest(unittest.TestCase):
  def setUp(self):
    self.frozendict = dedupe.core.frozendict
    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    self.deduper = dedupe.Dedupe(fields)
    self.wholeFieldPredicate = dedupe.predicates.wholeFieldPredicate
    self.sameThreeCharStartPredicate = dedupe.predicates.sameThreeCharStartPredicate
    self.training_pairs = {
        0: [(self.frozendict({"name": "Bob", "age": "50"}),
             self.frozendict({"name": "Charlie", "age": "75"})),
            (self.frozendict({"name": "Meredith", "age": "40"}),
             self.frozendict({"name": "Sue", "age": "10"}))], 
        1: [(self.frozendict({"name": "Jimmy", "age": "20"}),
             self.frozendict({"name": "Jimbo", "age": "21"})),
            (self.frozendict({"name": "Willy", "age": "35"}),
             self.frozendict({"name": "William", "age": "35"}))]
      }
    self.predicate_functions = (self.wholeFieldPredicate, self.sameThreeCharStartPredicate)
    
class PredicatesTest(unittest.TestCase):
  def test_predicates_correctness(self):
    field = '123 16th st'
    assert dedupe.predicates.wholeFieldPredicate(field) == ('123 16th st',)
    assert dedupe.predicates.tokenFieldPredicate(field) == ('123', '16th', 'st')
    assert dedupe.predicates.commonIntegerPredicate(field) == ('123', '16')
    assert dedupe.predicates.sameThreeCharStartPredicate(field) == ('123',)
    assert dedupe.predicates.sameFiveCharStartPredicate(field) == ('123 1',)
    assert dedupe.predicates.sameSevenCharStartPredicate(field) == ('123 16t',)
    assert dedupe.predicates.nearIntegersPredicate(field) == (15, 16, 17, 122, 123, 124)
    assert dedupe.predicates.commonFourGram(field) == ('123 ', '23 1', '3 16', ' 16t', '16th', '6th ', 'th s', 'h st')
    assert dedupe.predicates.commonSixGram(field) == ('123 16', '23 16t', '3 16th', ' 16th ', '16th s', '6th st')

class FieldDistances(unittest.TestCase):
  def test_field_distance_simple(self) :
    fieldDistances = dedupe.core.fieldDistances
    deduper = dedupe.Dedupe({'name' : {'type' :'String'},
                             'source' : {'type' : 'Source',
                                         'Source Names' : ['a', 'b']}})

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
                             })

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
                             })

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
         numpy.array([[ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.],
                      [ 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.]]),
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
                           })

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

