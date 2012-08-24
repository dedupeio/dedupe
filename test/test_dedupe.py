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
    
  def test_chaudhi_neighbor_list(self):
    neighborDict = dedupe.clustering.chaudhi.neighborDict
    assert neighborDict(self.dupes) == {1: [(2, 0.14), (3, 0.28), (5, 0.4), (4, 0.8)], 
                                        2: [(1, 0.14), (3, 0.14), (5, 0.28), (4, 0.8)],    
                                        3: [(2, 0.14), (1, 0.28), (5, 0.5), (4, 0.7)], 
                                        4: [(5, 0.28), (3, 0.7), (1, 0.8), (2, 0.8)], 
                                        5: [(2, 0.28), (4, 0.28), (1, 0.4), (3, 0.5)]
                                      }
    
  def test_chaudhi(self):
    chaudhi = dedupe.clustering.chaudhi.cluster
    assert chaudhi(self.dupes, 0, 6, 2) == []
    assert chaudhi(self.dupes, 1000, 6, 2) == [set([1, 2, 3, 4, 5])]
    print chaudhi(self.dupes, 3, 6, 2)
    
    
    #assert chaudhi(self.dupes, 0.5) == [set([1, 2])]
    #assert chaudhi(self.dupes, 1) == [set([1, 2, 3, 4, 5, 7])]
        
if __name__ == "__main__":
    unittest.main()