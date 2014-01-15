import dedupe
import unittest
import numpy
import random
import itertools
import multiprocessing
import collections

DATA = {  100 : {"name": "Bob", "age": "50"},
          105 : {"name": "Charlie", "age": "75"},
          110 : {"name": "Meredith", "age": "40"},
          115 : {"name": "Sue", "age": "10"}, 
          120 : {"name": "Jimmy", "age": "20"},
          125 : {"name": "Jimbo", "age": "21"},
          130 : {"name": "Willy", "age": "35"},
          135 : {"name": "William", "age": "35"},
          140 : {"name": "Martha", "age": "19"},
          145 : {"name": "Kyle", "age": "27"}
        }

DATA_SAMPLE = ((dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '50', 'name': 'Bob'})),
               (dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '35', 'name': 'William'})),
               (dedupe.core.frozendict({'age': '10', 'name': 'Sue'}), 
                dedupe.core.frozendict({'age': '35', 'name': 'William'})),
               (dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '20', 'name': 'Jimmy'})),
               (dedupe.core.frozendict({'age': '75', 'name': 'Charlie'}), 
                dedupe.core.frozendict({'age': '21', 'name': 'Jimbo'})))




class ConvenienceTest(unittest.TestCase):
  def test_data_sample(self):
    random.seed(123)
    numpy.random.seed(123)
    assert dedupe.dataSample(DATA ,5) == \
      (({'age': '27', 'name': 'Kyle'}, 
        {'age': '50', 'name': 'Bob'}), 
       ({'age': '50', 'name': 'Bob'}, 
        {'age': '21', 'name': 'Jimbo'}), 
       ({'age': '35', 'name': 'William'}, 
        {'age': '40', 'name': 'Meredith'}), 
       ({'age': '20', 'name': 'Jimmy'}, 
        {'age': '40', 'name': 'Meredith'}), 
       ({'age': '10', 'name': 'Sue'}, 
        {'age': '50', 'name': 'Bob'}))


class SourceComparatorTest(unittest.TestCase) :
  def test_comparator(self) :
    deduper = dedupe.Dedupe({'name' : {'type' : 'Source',
                                       'Source Names' : ['a', 'b'],
                                       'Has Missing' : True}}, ())

    source_comparator = deduper.data_model['fields']['name']['comparator']
    assert source_comparator('a', 'a') == 0
    assert source_comparator('b', 'b') == 1
    assert source_comparator('a', 'b') == 2
    assert source_comparator('b', 'a') == 2
    self.assertRaises(ValueError, source_comparator, 'b', 'c')
    self.assertRaises(ValueError, source_comparator, '', 'c')
    assert numpy.isnan(source_comparator('', 'b'))


class DataModelTest(unittest.TestCase) :

  def test_data_model(self) :
    OrderedDict = dedupe.backport.OrderedDict
    DataModel = dedupe.datamodel.DataModel
    from dedupe.distance.affinegap import normalizedAffineGapDistance
    from dedupe.distance.haversine import compareLatLong
    from dedupe.distance.jaccard import compareJaccard
    
    self.assertRaises(TypeError, DataModel)
    assert DataModel({}) == {'fields': OrderedDict(), 'bias': 0}
    self.assertRaises(ValueError, DataModel, {'a' : 'String'})
    self.assertRaises(ValueError, DataModel, {'a' : {'foo' : 'bar'}})
    self.assertRaises(ValueError, DataModel, {'a' : {'type' : 'bar'}})
    self.assertRaises(ValueError, DataModel, {'a-b' : {'type' : 'Interaction'}})
    self.assertRaises(ValueError, DataModel, {'a-b' : {'type' : 'Custom'}})
    self.assertRaises(ValueError, DataModel, {'a-b' : {'type' : 'String', 'comparator' : 'foo'}})

    self.assertRaises(KeyError, DataModel, {'a-b' : {'type' : 'Interaction',
                                                           'Interaction Fields' : ['a', 'b']}})
    assert DataModel({'a' : {'type' : 'String'}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance})]),
       'bias': 0}
    assert DataModel({'a' : {'type' : 'LatLong'}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'LatLong', 
                                     'comparator': compareLatLong})]), 
       'bias': 0}
    assert DataModel({'a' : {'type' : 'Set'}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'Set', 
                                     'comparator': compareJaccard})]), 
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String', 'Has Missing' : True}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': True, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance}), 
                              ('a: not_missing', {'type': 'Missing Data'})]), 
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String', 'Has Missing' : False}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance})]),
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String'}, 'b' : {'type' : 'String'}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator' : normalizedAffineGapDistance}), 
                              ('b', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance})]),
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String'}, 
                      'b' : {'type' : 'String'},
                      'a-b' : {'type' : 'Interaction', 
                               'Interaction Fields' : ['a', 'b']}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance}), 
                               ('b', {'Has Missing': False, 
                                      'type': 'String', 
                                      'comparator': normalizedAffineGapDistance}), 
                               ('a-b', {'Has Missing': False, 
                                        'type': 'Interaction', 
                                        'Interaction Fields': ['a', 'b']})]), 
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String', 'Has Missing' : True}, 
                      'b' : {'type' : 'String'},
                      'a-b' : {'type' : 'Interaction', 
                               'Interaction Fields' : ['a', 'b']}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': True, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance}), 
                               ('b', {'Has Missing': False, 
                                      'type': 'String', 
                                      'comparator': normalizedAffineGapDistance}), 
                               ('a-b', {'Has Missing': True, 
                                        'type': 'Interaction', 
                                        'Interaction Fields': ['a', 'b']}),
                              ('a: not_missing', {'type': 'Missing Data'}), 
                              ('a-b: not_missing', {'type': 'Missing Data'})]), 
       'bias': 0}
    assert DataModel({'a' : {'type' : 'String', 'Has Missing' : False}, 
                      'b' : {'type' : 'String'},
                      'a-b' : {'type' : 'Interaction', 
                               'Interaction Fields' : ['a', 'b']}}) == \
      {'fields': OrderedDict([('a', {'Has Missing': False, 
                                     'type': 'String', 
                                     'comparator': normalizedAffineGapDistance}), 
                               ('b', {'Has Missing': False, 
                                      'type': 'String', 
                                      'comparator': normalizedAffineGapDistance}), 
                               ('a-b', {'Has Missing': False, 
                                        'type': 'Interaction', 
                                        'Interaction Fields': ['a', 'b']})]),
       'bias': 0}

class DedupeInitializeTest(unittest.TestCase) :
  def test_initialize_fields(self) :
    self.assertRaises(TypeError, dedupe.Dedupe)
    self.assertRaises(TypeError, dedupe.Dedupe, [])

    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    deduper = dedupe.Dedupe(fields, [])

    assert deduper.matches is None
    assert deduper.blocker is None


  def test_base_predicates(self) :
    deduper = dedupe.Dedupe({'name' : {'type' : 'String'}}, [])
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

    assert deduper.blockerTypes() == {'String' : string_predicates + tfidf_string_predicates}


class DedupeClassTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123) 
    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    data_sample = DATA_SAMPLE
    self.deduper = dedupe.Dedupe(fields, data_sample)

  def test_blockPairs(self) :
    self.assertRaises(ValueError, self.deduper.blockedPairs, ((),))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({1:2},))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({'name':'Frank', 'age':21},))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({'1' : {'name' : 'Frank',
                                                                      'height' : 72}},))
    assert [] == list(self.deduper.blockedPairs(({'1' : {'name' : 'Frank',
                                                         'age' : 72}},)))
    assert list(self.deduper.blockedPairs(({'1' : {'name' : 'Frank',
                                                   'age' : 72},
                                            '2' : {'name' : 'Bob',
                                                   'age' : 27}},))) == \
                  [(('1', {'age': 72, 'name': 'Frank'}), 
                    ('2', {'age': 27, 'name': 'Bob'}))]

                                    

  def test_add_training(self) :
    training_pairs = {'distinct' : self.deduper.data_sample[0:3],
                      'match' : self.deduper.data_sample[3:6]}
    self.deduper._addTrainingData(training_pairs)
    numpy.testing.assert_equal(self.deduper.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match'])
    numpy.testing.assert_almost_equal(self.deduper.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]),
                                      4)
    self.deduper._addTrainingData(training_pairs)
    numpy.testing.assert_equal(self.deduper.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match']*2)

    numpy.testing.assert_almost_equal(self.deduper.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]*2),
                                      4)





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
    self.dupes = numpy.array([((1,2), .86),
                              ((1,3), .72),
                              ((1,4), .2),
                              ((1,5), .6),                 
                              ((2,3), .86),
                              ((2,4), .2),
                              ((2,5), .72),
                              ((3,4), .3),
                              ((3,5), .5),
                              ((4,5), .72)],
                             dtype = [('pairs', 'i4', 2), 
                                      ('score', 'f4', 1)])

    #Dupes with Ids as String
    self.str_dupes = numpy.array([(('1', '2'), .86),
                                  (('1', '3'), .72),
                                  (('1', '4'), .2),
                                  (('1', '5'), .6),
                                  (('2', '3'), .86),
                                  (('2', '4'), .2),
                                  (('2', '5'), .72),
                                  (('3', '4'), .3),
                                  (('3', '5'), .5),
                                  (('4', '5'), .72)],
                                 dtype = [('pairs', 'S4', 2), ('score', 'f4', 1)])

    self.bipartite_dupes = (((1,5), .1),
                            ((1,6), .72),
                            ((1,7), .2),
                            ((1,8), .6),
                            ((2,5), .2),
                            ((2,6), .2),
                            ((2,7), .72),
                            ((2,8), .3),
                            ((3,5), .24),
                            ((3,6), .72),
                            ((3,7), .24),
                            ((3,8), .65),
                            ((4,5), .63),
                            ((4,6), .96),
                            ((4,7), .23),
                            ((4,8), .74))


  def test_hierarchical(self):
    hierarchical = dedupe.clustering.cluster
    assert hierarchical(self.dupes, 1) == []
    assert hierarchical(self.dupes, 0.5) == [set([1, 2, 3]), set([4,5])]
    assert hierarchical(self.dupes, 0) == [set([1, 2, 3, 4, 5])]
    assert hierarchical(self.str_dupes, 1) == []
    assert hierarchical(self.str_dupes, 0.5) == [set(['1', '2', '3']), 
                                                      set(['4','5'])]
    assert hierarchical(self.str_dupes, 0) == [set(['1', '2', '3', '4', '5'])]

  def test_greedy_matching(self):
    greedyMatch = dedupe.clustering.greedyMatching
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0.5) == [(4, 6), 
                                          (2, 7),
                                          (3, 8)]
    
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0) == [(4, 6), 
                                        (2, 7),
                                        (3, 8), 
                                        (1, 5)]
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0.8) == [(4, 6)]
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=1) == []


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
    
class TfidfTest(unittest.TestCase):
  def setUp(self):
    self.field = "Hello World world"
    self.record_id = 20
    self.data_d = {
                     100 : {"name": "Bob", "age": "50", "dataset": 0},
                     105 : {"name": "Charlie", "age": "75", "dataset": 1},
                     110 : {"name": "Meredith", "age": "40", "dataset": 1},
                     115 : {"name": "Sue", "age": "10", "dataset": 0},
                     120 : {"name": "Jimbo", "age": "21","dataset": 1},
                     125 : {"name": "Jimbo", "age": "21", "dataset": 0},
                     130 : {"name": "Willy", "age": "35", "dataset": 0},
                     135 : {"name": "Willy", "age": "35", "dataset": 1},
                     140 : {"name": "Martha", "age": "19", "dataset": 1},
                     145 : {"name": "Kyle", "age": "27", "dataset": 0},
                  }
    
    self.tfidf_fields = ["name"]



  def test_unconstrained_inverted_index(self):

    blocker = dedupe.blocking.DedupeBlocker()
    blocker.tfidf_fields = {"name" : [dedupe.tfidf.TfidfPredicate(0.0)]}

    blocker.tfIdfBlock(((record_id, record["name"]) 
                        for record_id, record 
                        in self.data_d.iteritems()),
                       "name")

    canopy = blocker.canopies.values()[0]

    assert canopy == {120: 120, 130: 130, 125: 120, 135: 130}

  def test_constrained_inverted_index(self):

    blocker = dedupe.blocking.RecordLinkBlocker()
    blocker.tfidf_fields = {"name" : [dedupe.tfidf.TfidfPredicate(0.0)]}

    fields_1 = dict((record_id, record["name"]) 
                    for record_id, record 
                    in self.data_d.iteritems()
                    if record["dataset"] == 0)

    fields_2 = dict((record_id, record["name"]) 
                    for record_id, record 
                    in self.data_d.iteritems()
                    if record["dataset"] == 1)

    blocker.tfIdfBlock(fields_1.items(), fields_2.items(), "name")

    canopy = blocker.canopies.values()[0]

    assert set(canopy.values()) <= set(fields_1.keys())

    assert canopy == {120: 125, 135: 130, 130: 130, 125: 125}



class PredicatesTest(unittest.TestCase):
  def test_predicates_correctness(self):
    field = '123 16th st'
    assert dedupe.predicates.wholeFieldPredicate(field) == ('123 16th st',)
    assert dedupe.predicates.tokenFieldPredicate(field) == ('123', '16th', 'st')
    assert dedupe.predicates.commonIntegerPredicate(field) == ('123', '16')
    assert dedupe.predicates.sameThreeCharStartPredicate(field) == ('123',)
    assert dedupe.predicates.sameFiveCharStartPredicate(field) == ('123 1',)
    assert dedupe.predicates.sameSevenCharStartPredicate(field) == ('123 16t',)
    assert dedupe.predicates.nearIntegersPredicate(field) == ('15', '17', '16', '122', '123', '124')
    assert dedupe.predicates.commonFourGram(field) == ('123 ', '23 1', '3 16', ' 16t', '16th', '6th ', 'th s', 'h st')
    assert dedupe.predicates.commonSixGram(field) == ('123 16', '23 16t', '3 16th', ' 16th ', '16th s', '6th st')
    assert dedupe.predicates.initials(field,12) == ()
    assert dedupe.predicates.initials(field,7) == ('123 16t',)
    assert dedupe.predicates.ngrams(field,3) == ('123','23 ','3 1',' 16','16t','6th','th ','h s',' st')






if __name__ == "__main__":
    unittest.main()

