import dedupe
import unittest

class AffineGapTest(unittest.TestCase):
  def setUp(self):
    self.affineGapDistance = dedupe.affinegap.affineGapDistance
    self.normalizedAffineGapDistance = dedupe.affinegap.normalizedAffineGapDistance
    
  def test_affine_gap_correctness(self):
    assert self.affineGapDistance('a', 'b', -5, 5, 5, 1) == 5
    assert self.affineGapDistance('ab', 'cd', -5, 5, 5, 1) == 10
    assert self.affineGapDistance('ab', 'cde', -5, 5, 5, 1) == 13
    assert self.affineGapDistance('a', 'cde', -5, 5, 5, 1) == 8.5
    assert self.affineGapDistance('a', 'cd', -5, 5, 5, 1) == 8
    assert self.affineGapDistance('b', 'a', -5, 5, 5, 1) == 5
    assert self.affineGapDistance('a', 'a', -5, 5, 5, 1) == -5
    assert self.affineGapDistance('a', '', -5, 5, 5, 1) == 3
    assert self.affineGapDistance('', '', -5, 5, 5, 1) == 0
    assert self.affineGapDistance('aba', 'aaa', -5, 5, 5, 1) == -5
    assert self.affineGapDistance('aaa', 'aba', -5, 5, 5, 1) == -5
    assert self.affineGapDistance('aaa', 'aa', -5, 5, 5, 1) == -7
    assert self.affineGapDistance('aaa', 'a', -5, 5, 5, 1) == -1.5
    assert self.affineGapDistance('aaa', '', -5, 5, 5, 1) == 4
    assert self.affineGapDistance('aaa', 'abba', -5, 5, 5, 1) == 1
    
  def test_normalized_affine_gap_correctness(self):
    assert self.normalizedAffineGapDistance('', '', -5, 5, 5, 1) == 0
    
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
    assert hierarchical(self.dupes, 0) == []
    assert hierarchical(self.dupes, 0.5) == [set([1, 2])]
    assert hierarchical(self.dupes, 1) == [set([1, 2, 3, 4, 5])]
    
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

    
 
        
if __name__ == "__main__":
    unittest.main()
