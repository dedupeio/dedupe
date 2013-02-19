import dedupe
import unittest

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
    assert self.affineGapDistance('a', '', -5, 5, 5, 1, 0.5) == 3
    assert self.affineGapDistance('', '', -5, 5, 5, 1, 0.5) == 0
    assert self.affineGapDistance('aba', 'aaa', -5, 5, 5, 1, 0.5) == -5
    assert self.affineGapDistance('aaa', 'aba', -5, 5, 5, 1, 0.5) == -5
    assert self.affineGapDistance('aaa', 'aa', -5, 5, 5, 1, 0.5) == -7
    assert self.affineGapDistance('aaa', 'a', -5, 5, 5, 1, 0.5) == -1.5
    assert self.affineGapDistance('aaa', '', -5, 5, 5, 1, 0.5) == 4
    assert self.affineGapDistance('aaa', 'abba', -5, 5, 5, 1, 0.5) == 1
    
  def test_normalized_affine_gap_correctness(self):
    assert self.normalizedAffineGapDistance('', '', -5, 5, 5, 1, 0.5) == 0
    
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
            
  def test_hierarchical(self):
    hierarchical = dedupe.clustering.hierarchical.cluster
    assert hierarchical(self.dupes, 1) == []
    assert hierarchical(self.dupes, 0.5) == [set([1, 2, 3]), set([4,5])]
    assert hierarchical(self.dupes, 0) == [set([1, 2, 3, 4, 5])]


  def test_chaudhuri_neighbor_list(self):
    neighborDict = dedupe.clustering.chaudhuri.neighborDict
    assert neighborDict(self.dupes) == {1: [(1, 0), (2, 0.14), (3, 0.28), (5, 0.4), (4, 0.8)], 
                                        2: [(2, 0), (1, 0.14), (3, 0.14), (5, 0.28), (4, 0.8)],    
                                        3: [(3, 0), (2, 0.14), (1, 0.28), (5, 0.5), (4, 0.7)], 
                                        4: [(4, 0), (5, 0.28), (3, 0.7), (1, 0.8), (2, 0.8)], 
                                        5: [(5, 0), (2, 0.28), (4, 0.28), (1, 0.4), (3, 0.5)]
                                      }

  

  def test_chaudhuri_neighbor_growth(self) :
    neighborhoodGrowth = dedupe.clustering.chaudhuri.neighborhoodGrowth
    neighbor_list = [0.10, 0.15, 0.20, 0.25] 
    assert neighborhoodGrowth(neighbor_list, 2) == 3

  def test_chaudhuri_compact_pairs(self) :
    compactPairs = dedupe.clustering.chaudhuri.compactPairs
    neighbors = dedupe.clustering.chaudhuri.neighborDict(self.dupes)
    assert compactPairs(neighbors, 2) == [((1, 2), [True, True, True, True], (2, 3)),
                                          ((1, 3), [False, True, True, True], (2, 2)),
                                          ((1, 4), [False, False, False, True], (2, 1)),
                                          ((1, 5), [False, False, False, True], (2, 4)),
                                          ((2, 3), [False, True, True, True], (3, 2)),
                                          ((2, 4), [False, False, False, True], (3, 1)),
                                          ((2, 5), [False, False, False, True], (3, 4)),
                                          ((3, 4), [False, False, False, True], (2, 1)),
                                          ((3, 5), [False, False, False, True], (2, 4)),
                                          ((4, 5), [False, False, False, True], (1, 4))]


  def test_chaudhuri_partition(self) :
    partition = dedupe.clustering.chaudhuri.partition
    neighbors = dedupe.clustering.chaudhuri.neighborDict(self.dupes)
    compact_pairs = dedupe.clustering.chaudhuri.compactPairs(neighbors, 2)
    assert partition(compact_pairs, 3) == [set([1, 2, 3])]


  def test_chaudhuri_sparseness_k_overlap(self) :
    kOverlap = dedupe.clustering.chaudhuri.kOverlap
    assert kOverlap([1,2,3,4,5], [1,2,3,4,5]) == [True, True, True, True, True]
    assert kOverlap([5,4,3,2,1], [1,2,3,4,5]) == [False, False, False, False, True]
    assert kOverlap([6,7,8,9,10], [1,2,3,4,5]) == [False, False, False, False, False]
    assert kOverlap([1,2,3,9,8], [1,2,3,4,5]) == [True, True, True, False, False]
  def test_chaudhuri(self):
    chaudhuri = dedupe.clustering.chaudhuri.cluster
    assert chaudhuri(self.dupes, 0, 6, 2) == []
    assert chaudhuri(self.dupes, 1000, 6, 2) == [set([1, 2, 3, 4, 5])]
    assert chaudhuri(self.dupes, 3, 6, 2) == [set([1, 2, 3])]

  def test_chaudhuri_growth_distribution(self) :
    growthDistrbutions = dedupe.clustering.chaudhuri.growthDistributions
    neighbors =  dedupe.clustering.chaudhuri.neighborDict(self.dupes)
    assert growthDistrbutions(neighbors, 2) == ([(1.0/5, 1),
                                                 (2.0/5, 2),
                                                 (1.0/5, 3),
                                                 (1.0/5, 4)],
                                                [(1.0/5, 1),
                                                 ((1.0/5 + 2.0/5), 2),
                                                 (4.0/5, 3),
                                                 (5.0/5, 4)])

  def test_chaudhuri_sparseness_threshold(self) :
    sparsenessThreshold = dedupe.clustering.chaudhuri.sparsenessThreshold
    neighbors =  dedupe.clustering.chaudhuri.neighborDict(self.dupes)
    assert sparsenessThreshold(neighbors, .1) == 2
    assert sparsenessThreshold(neighbors, .2) == 3
    assert sparsenessThreshold(neighbors, .3) == 3
    assert sparsenessThreshold(neighbors, .4) == 3
    assert sparsenessThreshold(neighbors, .5) == 3
    assert sparsenessThreshold(neighbors, .6) == 4

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
    
  def test_initializer(self) :
    (training_dupes,
     training_distinct,
     predicate_set,
     _overlap) =  dedupe.blocking._initializeTraining(self.training_pairs,
                                                      self.deduper.data_model,
                                                      self.predicate_functions,
                                                      [],
                                                      {})

    assert training_dupes == [(self.frozendict({'age': '20', 'name': 'Jimmy'}),
                               self.frozendict({'age': '21', 'name': 'Jimbo'})),
                              (self.frozendict({'age': '35', 'name': 'Willy'}),
                               self.frozendict({'age': '35', 'name': 'William'}))]
    assert training_distinct == [(self.frozendict({'age': '50', 'name': 'Bob'}),
                                  self.frozendict({'age': '75', 'name': 'Charlie'})),
                                 (self.frozendict({'age': '40', 'name': 'Meredith'}),
                                  self.frozendict({'age': '10', 'name': 'Sue'}))]
    assert predicate_set == [((self.wholeFieldPredicate, 'age'),),
                             ((self.wholeFieldPredicate, 'name'),),
                             ((self.sameThreeCharStartPredicate, 'age'),),
                             ((self.sameThreeCharStartPredicate, 'name'),),
                             ((self.wholeFieldPredicate, 'age'),
                              (self.wholeFieldPredicate, 'name')),
                             ((self.wholeFieldPredicate, 'age'),
                              (self.sameThreeCharStartPredicate, 'name')),
                             ((self.wholeFieldPredicate, 'name'),
                              (self.sameThreeCharStartPredicate, 'age')),
                             ((self.sameThreeCharStartPredicate, 'age'),
                              (self.sameThreeCharStartPredicate, 'name'))]

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
        
if __name__ == "__main__":
    unittest.main()
