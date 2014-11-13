# -*- coding: utf-8 -*-

import dedupe
import unittest
import numpy
import random
import itertools
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





class SourceComparatorTest(unittest.TestCase) :
  def test_comparator(self) :
    deduper = dedupe.Dedupe([{'field' :'name', 'type' : 'Source',
                              'sources' : ['a', 'b'],
                              'has missing' : True}], ())

    source_comparator = deduper.data_model['fields'][0].comparator
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
    
    self.assertRaises(TypeError, DataModel)
    assert DataModel({}) == {'fields': [], 'bias': 0}

    data_model = DataModel([{'field' : 'a', 
                            'variable name' : 'a', 
                            'type' : 'String'}, 
                            {'field' : 'b', 
                             'variable name' : 'b',
                             'type' : 'String'},
                            {'type' : 'Interaction', 
                             'interaction variables' : ['a', 'b']}])

    assert data_model['fields'][2].interaction_fields  == ['a', 'b']

    data_model = DataModel([{'field' : 'a', 
                             'variable name' : 'a', 
                             'type' : 'String',
                             'has missing' : True}, 
                            {'field' : 'b', 
                             'variable name' : 'b',
                             'type' : 'String'},
                            {'type' : 'Interaction', 
                             'interaction variables' : ['a', 'b']}])

    #print data_model['fields']
    assert data_model['fields'][2].has_missing == True

    data_model = DataModel([{'field' : 'a', 
                             'variable name' : 'a', 
                             'type' : 'String',
                             'has missing' : False}, 
                            {'field' : 'b', 
                             'variable name' : 'b',
                             'type' : 'String'},
                            {'type' : 'Interaction', 
                             'interaction variables' : ['a', 'b']}])


    assert data_model['fields'][2].has_missing == False





class AffineGapTest(unittest.TestCase):
  def setUp(self):
    self.affineGapDistance = dedupe.distance.affinegap.affineGapDistance
    self.normalizedAffineGapDistance = dedupe.distance.affinegap.normalizedAffineGapDistance
    
  def test_affine_gap_correctness(self):
    assert self.affineGapDistance('a', u'b', -5, 5, 5, 1, 0.5) == 5
    assert self.affineGapDistance('ab', 'cd', -5, 5, 5, 1, 0.5) == 10
    assert self.affineGapDistance('ab', 'cde', -5, 5, 5, 1, 0.5) == 13
    assert self.affineGapDistance('ab', u'cdë', -5, 5, 5, 1, 0.5) == 13
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
    

class ConnectedComponentsTest(unittest.TestCase) :
  def test_components(self) :
    G = numpy.array([((1, 2), .1),
                     ((2, 3), .2),
                     ((4, 5), .2),
                     ((4, 6), .2),
                     ((7, 9), .2),
                     ((8, 9), .2),
                     ((10, 11), .2),
                     ((12, 13), .2),
                     ((12, 14), .5),
                     ((11, 12), .2)],
                    dtype = [('pairs', 'i4', 2), ('score', 'f4', 1)])
    components = dedupe.clustering.connected_components
    numpy.testing.assert_equal(list(components(G, 30000)), \
                               [numpy.array([([1, 2], 0.10000000149011612), 
                                             ([2, 3], 0.20000000298023224)], 
                                            dtype=[('pairs', 'i4', (2,)), 
                                                   ('score', '<f4')]), 
                                numpy.array([([4, 5], 0.20000000298023224), 
                                             ([4, 6], 0.20000000298023224)], 
                                            dtype=[('pairs', 'i4', (2,)), 
                                                   ('score', '<f4')]), 
                                numpy.array([([12, 13], 0.20000000298023224), 
                                             ([12, 14], 0.5),
                                             ([10, 11], 0.20000000298023224), 
                                             ([11, 12], 0.20000000298023224)],
                                            dtype=[('pairs', 'i4', (2,)), 
                                                   ('score', '<f4')]), 
                                numpy.array([([7, 9], 0.20000000298023224), 
                                             ([8, 9], 0.20000000298023224)], 
                                            dtype=[('pairs', 'i4', (2,)), 
                                                   ('score', '<f4')])])

  

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
                              ((4,5), .72),
                              ((10,11), .9)],
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
                            ((5,8), .24))

  def clusterEquals(self, x, y) :
    for cluster_a, cluster_b in zip(x, y) :
      if cluster_a[0] != cluster_b[0] :
        return False
      for score_a, score_b in zip(cluster_a[1], cluster_b[1]) :
        if round(score_a, 2) != round(score_b, 2) :
          return False
      else :
        return True



  def test_hierarchical(self):
    hierarchical = dedupe.clustering.cluster
    assert self.clusterEquals(hierarchical(self.dupes, 1),
                              [((10, 11),  
                                (0.899,
                                 0.899))])

    assert self.clusterEquals(hierarchical(self.dupes, 0.5),
                              [((1, 2, 3), 
                                (0.790, 
                                 0.860, 
                                 0.790)), 
                               ((4, 5), 
                                (0.720, 
                                 0.720)), 
                               ((10, 11), 
                                (0.899, 
                                 0.899))])

    assert self.clusterEquals(hierarchical(self.dupes, 0),
                              [((1, 2, 3, 4, 5), 
                                (0.5950000137090683, 
                                 0.66000001132488251, 
                                 0.59500001370, 
                                 0.3550000041, 
                                 0.635)), 
                               ((10, 11), 
                                (0.899, 
                                 0.899))]
)

    assert hierarchical(self.str_dupes, 1) == []
    assert zip(*hierarchical(self.str_dupes, 0.5))[0] == (('1', '2', '3'), 
                                                          ('4','5'))
    assert zip(*hierarchical(self.str_dupes, 0))[0] == (('1', '2', '3', '4', '5'),)


  def test_greedy_matching(self):
    greedyMatch = dedupe.clustering.greedyMatching
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0.5) == [((4, 6), 0.96), 
                                          ((2, 7), 0.72), 
                                          ((3, 8), 0.65)]
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0) == [((4, 6), 0.96), 
                                        ((2, 7), 0.72), 
                                        ((3, 8), 0.65), 
                                        ((1, 5), 0.1)]
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=0.8) == [((4, 6), 0.96)]
    assert greedyMatch(self.bipartite_dupes, 
                       threshold=1) == []

  def test_gazette_matching(self):
    gazetteMatch = dedupe.clustering.gazetteMatching

    assert set(gazetteMatch(self.bipartite_dupes, 
                            threshold=0.5)) == set([(((4, 6), 0.96),), 
                                                    (((1, 6), 0.72),), 
                                                    (((2, 7), 0.72),), 
                                                    (((3, 6), 0.72),)])

    assert set(gazetteMatch(self.bipartite_dupes, 
                            threshold=0, n=2)) == set([(((1, 6), 0.72), 
                                                        ((1, 8), 0.6)), 
                                                       (((2, 7), 0.72),
                                                        ((2, 8), 0.3)), 
                                                       (((3, 6), 0.72), 
                                                        ((3, 8), 0.65)), 
                                                       (((4, 6), 0.96), 
                                                        ((4, 5), 0.63)),
                                                       (((5, 8), 0.24),)])

    assert set(gazetteMatch(self.bipartite_dupes, 
                        threshold=0)) == set([(((4, 6), 0.96),), 
                                              (((1, 6), 0.72),), 
                                              (((2, 7), 0.72),), 
                                              (((3, 6), 0.72),), 
                                              (((5, 8), 0.24),)])

    assert gazetteMatch(self.bipartite_dupes, 
                        threshold=0.8) == [(((4,6), 0.96),)]

    assert gazetteMatch(self.bipartite_dupes, 
                        threshold=1) == []


class PredicatesTest(unittest.TestCase):
  def test_predicates_correctness(self):
    field = '123 16th st'
    assert dedupe.predicates.existsPredicate(field) == (1,)
    assert dedupe.predicates.existsPredicate('') == (0,)
    assert dedupe.predicates.existsPredicate(1) == (1,)
    assert dedupe.predicates.existsPredicate(0) == (0,)
    assert dedupe.predicates.sortedAcronym(field) == ('11s',)
    assert dedupe.predicates.wholeFieldPredicate('') == ()
    assert dedupe.predicates.wholeFieldPredicate(field) == ('123 16th st',)
    assert dedupe.predicates.firstTokenPredicate(field) == ('123',)
    assert dedupe.predicates.firstTokenPredicate('') == ()
    assert dedupe.predicates.firstTokenPredicate('123/') == ('123',)
    assert dedupe.predicates.tokenFieldPredicate(' ') == set([])
    assert dedupe.predicates.tokenFieldPredicate(field) == set(['123', '16th', 'st'])
    assert dedupe.predicates.commonIntegerPredicate(field) == set(['123', '16'])
    assert dedupe.predicates.commonIntegerPredicate('foo') == set([])
    assert dedupe.predicates.firstIntegerPredicate('foo') == ()
    assert dedupe.predicates.firstIntegerPredicate('1foo') == ('1',)
    assert dedupe.predicates.firstIntegerPredicate('f1oo') == ()
    assert dedupe.predicates.sameThreeCharStartPredicate(field) == ('123',)
    assert dedupe.predicates.sameThreeCharStartPredicate('12') == ()
    assert dedupe.predicates.commonFourGram('12') == set([])
    assert dedupe.predicates.sameFiveCharStartPredicate(field) == ('123 1',)
    assert dedupe.predicates.sameSevenCharStartPredicate(field) == ('123 16t',)
    assert dedupe.predicates.nearIntegersPredicate(field) == set(['15', '17', '16', '122', '123', '124'])
    assert dedupe.predicates.commonFourGram(field) == set(['123 ', '23 1', '3 16', ' 16t', '16th', '6th ', 'th s', 'h st'])
    assert dedupe.predicates.commonSixGram(field) == set(['123 16', '23 16t', '3 16th', ' 16th ', '16th s', '6th st'])
    assert dedupe.predicates.initials(field,12) == ()
    assert dedupe.predicates.initials(field,7) == ('123 16t',)
    assert dedupe.predicates.ngrams(field,3) == set(['123','23 ','3 1',' 16','16t','6th','th ','h s',' st'])
    assert dedupe.predicates.commonTwoElementsPredicate((1,2,3)) == set(('1 2','2 3'))
    assert dedupe.predicates.commonTwoElementsPredicate((1,)) == set([])
    assert dedupe.predicates.commonThreeElementsPredicate((1,2,3)) == set(('1 2 3',))
    assert dedupe.predicates.commonThreeElementsPredicate((1,)) == set([])
    assert dedupe.predicates.fingerprint('time sandwich') == (u'sandwichtime',)
    assert dedupe.predicates.oneGramFingerprint('sandwich time') == (u'acdehimnstw',)
    assert dedupe.predicates.twoGramFingerprint('sandwich time') == (u'anchdwhicimmendsattiwi',)




if __name__ == "__main__":
    unittest.main()

