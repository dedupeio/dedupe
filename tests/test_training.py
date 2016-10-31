import dedupe
import unittest

class TrainingTest(unittest.TestCase):
    def setUp(self):

        field_definition = [{'field' : 'name', 'type': 'String'}]
        self.data_model = dedupe.Dedupe(field_definition).data_model
        self.training_pairs = {
            'match': [({"name": "Bob", "age": "50"},
                       {"name": "Bob", "age": "75"}),
                      ({"name": "Meredith", "age": "40"},
                       {"name": "Sue", "age": "10"})], 
            'distinct': [({"name": "Jimmy", "age": "20"},
                          {"name": "Jimbo", "age": "21"}),
                         ({"name": "Willy", "age": "35"},
                          {"name": "William", "age": "35"}),
                         ({"name": "William", "age": "36"},
                          {"name": "William", "age": "35"})]
        }
    
        self.training = self.training_pairs['match'] + self.training_pairs['distinct']
        self.training_records = []
        for pair in self.training:
          for record in pair:
            if record not in self.training_records:
              self.training_records.append(record)
    
        self.simple = lambda x : set([str(k) for k in x 
                                      if "CompoundPredicate" not in str(k)])
    
    
    def test_dedupe_coverage(self) :
        predicates = self.data_model.predicates()
        blocker = dedupe.blocking.Blocker(predicates)
        blocker.indexAll({i : x for i, x in enumerate(self.training_records)})
        coverage = dedupe.training.coveredPairs(blocker.predicates,
                                                self.training)
        assert self.simple(coverage.keys()).issuperset(
              set(["SimplePredicate: (tokenFieldPredicate, name)", 
                   "SimplePredicate: (commonSixGram, name)", 
                   "TfidfTextCanopyPredicate: (0.4, name)", 
                   "SimplePredicate: (sortedAcronym, name)",
                   "SimplePredicate: (sameThreeCharStartPredicate, name)", 
                   "TfidfTextCanopyPredicate: (0.2, name)", 
                   "SimplePredicate: (sameFiveCharStartPredicate, name)", 
                   "TfidfTextCanopyPredicate: (0.6, name)", 
                   "SimplePredicate: (wholeFieldPredicate, name)", 
                   "TfidfTextCanopyPredicate: (0.8, name)", 
                   "SimplePredicate: (commonFourGram, name)", 
                   "SimplePredicate: (firstTokenPredicate, name)", 
                   "SimplePredicate: (sameSevenCharStartPredicate, name)"]))


    def test_unique(self):
        target = ([{1: 1, 2: 2}, {3: 3, 4: 4}],
                  [{3: 3, 4: 4}, {1: 1, 2: 2}])
        
        assert dedupe.training.unique([{1: 1, 2: 2}, {3: 3, 4: 4}, {1: 1, 2: 2}]) in target


        

if __name__ == "__main__":
    unittest.main()
