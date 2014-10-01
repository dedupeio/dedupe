import dedupe
import unittest

class SemiSupervisedNonDuplicates(unittest.TestCase) :
    def setUp(self) :
        self.deduper = dedupe.Gazetteer([{'field' : 'name', 'type' : 'String'}])
        self.sSND = dedupe.training.semiSupervisedNonDuplicates
        
    def test_empty_sample(self) :

        assert len(list(self.sSND(self.deduper.data_sample, 
                                  self.deduper.data_model, 0.7, 2000))) == 0


if __name__ == "__main__":
    unittest.main()
