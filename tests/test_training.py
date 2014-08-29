import dedupe
import unittest

class SemiSupervisedNonDuplicates(unittest.TestCase) :
    def setUp(self) :
        deduper = dedupe.Dedupe([{'field' : 'name', 'type' : 'String'}])
        self.data_model = deduper.data_model
        self.sSND = dedupe.training.semiSupervisedNonDuplicates
        
    def test_empty_sample(self) :

        assert len(list(self.sSND([], self.data_model, 0.7, 2000))) == 0

if __name__ == "__main__":
    unittest.main()
