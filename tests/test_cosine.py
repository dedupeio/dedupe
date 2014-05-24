import unittest
<<<<<<< HEAD
from dedupe.distance.cosine import CosineSetSimilarity
import numpy


class TestCosineClass(unittest.TestCase):
=======
from dedupe.distance.cosine import CosineSetSimilarity, CosineTextSimilarity
import numpy


class TestSetCosineClass(unittest.TestCase):
    def setUp(self):
        self.ilist = [frozenset(['a', 'b', 'c']),
                      frozenset(['b', 'c', 'd']),
                      frozenset(['d', 'e', 'f'])
                      ]

    def test_cosine(self):
        cosine = CosineSetSimilarity(self.ilist)
        s1 = self.ilist[0]
        s2 = self.ilist[1]
        cosine_sim = cosine(s1, s2)
        self.assertAlmostEqual(cosine_sim, 0.378, places=3)

    def test_cosine_na(self):
        cosine = CosineSetSimilarity(self.ilist)
        cosine_sim = cosine(self.ilist[0], frozenset([]))
        assert numpy.isnan(cosine_sim)
        
    def test_cosine_identical(self):
        cosine = CosineSetSimilarity(self.ilist)
        cosine_sim = cosine(self.ilist[0], self.ilist[0])
        self.assertAlmostEqual(cosine_sim, 1, places=5)

class TestTextCosineClass(unittest.TestCase):
>>>>>>> add test for cosine text
    def setUp(self):
        self.ilist = ['a b c',
                      'b c d',
                      'd e f']

    def test_cosine(self):
<<<<<<< HEAD
        cosine = CosineSetSimilarity(self.ilist)
=======
        cosine = CosineTextSimilarity(self.ilist)
>>>>>>> add test for cosine text
        s1 = self.ilist[0]
        s2 = self.ilist[1]
        cosine_sim = cosine(s1, s2)
        self.assertAlmostEqual(cosine_sim, 0.378, places=3)

    def test_cosine_na(self):
<<<<<<< HEAD
        cosine = CosineSetSimilarity(self.ilist)
        cosine_sim = cosine(self.ilist[0], frozenset([]))
        assert numpy.isnan(cosine_sim)
        
    def test_cosine_identical(self):
        cosine = CosineSetSimilarity(self.ilist)
=======
        cosine = CosineTextSimilarity(self.ilist)
        cosine_sim = cosine(self.ilist[0], '')
        assert numpy.isnan(cosine_sim)
        
    def test_cosine_identical(self):
        cosine = CosineTextSimilarity(self.ilist)
>>>>>>> add test for cosine text
        cosine_sim = cosine(self.ilist[0], self.ilist[0])
        self.assertAlmostEqual(cosine_sim, 1, places=5)

    
if __name__ == '__main__':
    unittest.main()
