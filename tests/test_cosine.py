import unittest
from dedupe.distance.cosine import createCosineSimilarity, CosineSimilarity
import numpy


class TestCosine(unittest.TestCase):
    def setUp(self):
        self.ilist = [frozenset(['a', 'b', 'c']),
                      frozenset(['b', 'c', 'd']),
                      frozenset(['d', 'e', 'f'])
                      ]

    def test_cosine(self):
        cosine = createCosineSimilarity(self.ilist, 1.0)
        cosine_sim = cosine(self.ilist[0],
                            self.ilist[1]
                            )

        ## idf: a = 0.47712
        ##      b = 0.17609
        ##      c = 0.17609
        ##      d = 0.17609
        ##      e, f = 0.48
        ## cosine(ilist[0], ilist[1]):
        ## numer = 0.176^2 * 2 = 0.06202
        ## denom_a = sqrt(0.48^2 + 2 * 0.176^2) = 0.53820
        ## denom_b = sqrt(0.176^2 * 3) = 0.30484
        ## denom = 0.16406
        ## cosine = numer / denom = 0.3780
        self.assertAlmostEqual(cosine_sim, 0.378, places=3)

    def test_cosine_na(self):
        cosine = createCosineSimilarity(self.ilist, 1.0)
        cosine_sim = cosine(self.ilist[0], frozenset([]))
        self.assertAlmostEqual(cosine_sim, 0, places=5)

    def test_cosine_identical(self):
        cosine = createCosineSimilarity(self.ilist, 1.0)
        cosine_sim = cosine(self.ilist[0], self.ilist[0])
        self.assertAlmostEqual(cosine_sim, 1, places=5)

class TestCosineClass(unittest.TestCase):
    def setUp(self):
        self.ilist = [frozenset(['a', 'b', 'c']),
                      frozenset(['b', 'c', 'd']),
                      frozenset(['d', 'e', 'f'])
                      ]

    def test_cosine(self):
        cosine = CosineSimilarity(self.ilist, 1.0)
        s1 = self.ilist[0]
        s2 = self.ilist[1]
        cosine_sim = cosine(s1, s2)
        self.assertAlmostEqual(cosine_sim, 0.378, places=3)

    def test_cosine_na(self):
        cosine = CosineSimilarity(self.ilist, 1.0)
        cosine_sim = cosine(self.ilist[0], frozenset([]))
        self.assertAlmostEqual(cosine_sim, 0, places=5)
        
    def test_cosine_identical(self):
        cosine = CosineSimilarity(self.ilist, 1.0)
        cosine_sim = cosine(self.ilist[0], self.ilist[0])
        self.assertAlmostEqual(cosine_sim, 1, places=5)

    
if __name__ == '__main__':
    unittest.main()
