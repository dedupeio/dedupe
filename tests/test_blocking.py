import dedupe
from collections import defaultdict
import unittest

class BlockingTest(unittest.TestCase):
  def setUp(self):
    self.frozendict = dedupe.core.frozendict

    field_definition = [{'field' : 'name', 'type': 'String'}, 
                        {'field' :'age', 'type': 'String'}]
    self.data_model = dedupe.Dedupe(field_definition).data_model
    self.training_pairs = {
        0: [((1, self.frozendict({"name": "Bob", "age": "50"})),
             (2, self.frozendict({"name": "Bob", "age": "75"}))),
            ((3, self.frozendict({"name": "Meredith", "age": "40"})),
             (4, self.frozendict({"name": "Sue", "age": "10"})))], 
        1: [((5, self.frozendict({"name": "Jimmy", "age": "20"})),
             (6, self.frozendict({"name": "Jimbo", "age": "21"}))),
            ((7, self.frozendict({"name": "Willy", "age": "35"})),
             (8, self.frozendict({"name": "William", "age": "35"}))),
            ((9, self.frozendict({"name": "William", "age": "36"})),
             (8, self.frozendict({"name": "William", "age": "35"})))]
      }

    self.training = self.training_pairs[0] + self.training_pairs[1]
    self.distinct_ids = [tuple(sorted([pair[0][0], pair[1][0]]))
                         for pair in
                         self.training_pairs[0]]
    self.dupe_ids = [tuple(sorted([pair[0][0], pair[1][0]]))
                     for pair in
                     self.training_pairs[1]]

  def test_dedupe_coverage(self) :
    predicates = self.data_model['fields'][1].predicates
    coverage = dedupe.blocking.DedupeCoverage(predicates, self.training)
    assert set([str(k) for k in coverage.overlap.keys()]) ==\
          set(["SimplePredicate: (tokenFieldPredicate, name)", 
               "SimplePredicate: (commonSixGram, name)", 
               "TfidfPredicate: (0.4, name)", 
               "SimplePredicate: (sameThreeCharStartPredicate, name)", 
               "TfidfPredicate: (0.2, name)", 
               "SimplePredicate: (sameFiveCharStartPredicate, name)", 
               "TfidfPredicate: (0.6, name)", 
               "SimplePredicate: (wholeFieldPredicate, name)", 
               "TfidfPredicate: (0.8, name)", 
               "SimplePredicate: (commonFourGram, name)", 
               "SimplePredicate: (firstTokenPredicate, name)", 
               "SimplePredicate: (sameSevenCharStartPredicate, name)"])

    overlap = coverage.predicateCoverage(predicates, self.distinct_ids)
    assert set(str(k) for k in overlap.keys()) ==\
          set(["TfidfPredicate: (0.4, name)", 
               "TfidfPredicate: (0.6, name)", 
              "SimplePredicate: (wholeFieldPredicate, name)", 
               "SimplePredicate: (sameThreeCharStartPredicate, name)",
               "SimplePredicate: (tokenFieldPredicate, name)", 
               "TfidfPredicate: (0.8, name)", 
               "SimplePredicate: (firstTokenPredicate, name)", 
               "TfidfPredicate: (0.2, name)"])

    overlap = coverage.predicateCoverage(predicates, self.dupe_ids)

    assert set(str(k) for k in overlap.keys()) ==\
          set(["SimplePredicate: (tokenFieldPredicate, name)", 
               "SimplePredicate: (commonSixGram, name)", 
               "TfidfPredicate: (0.4, name)", 
               "SimplePredicate: (sameThreeCharStartPredicate, name)", 
               "TfidfPredicate: (0.2, name)", 
               "SimplePredicate: (sameFiveCharStartPredicate, name)", 
               "TfidfPredicate: (0.6, name)", 
               "SimplePredicate: (wholeFieldPredicate, name)", 
               "TfidfPredicate: (0.8, name)", 
               "SimplePredicate: (firstTokenPredicate, name)", 
               "SimplePredicate: (commonFourGram, name)", 
               "SimplePredicate: (sameSevenCharStartPredicate, name)"])

    predicates = self.data_model['fields'][1].predicates

    coverage = dedupe.blocking.RecordLinkCoverage(predicates, self.training)
    print coverage.overlap.keys()
    assert set([str(k) for k in coverage.overlap.keys()]) ==\
          set(["SimplePredicate: (tokenFieldPredicate, name)", 
               "SimplePredicate: (commonSixGram, name)", 
               "TfidfPredicate: (0.4, name)", 
               "SimplePredicate: (sameThreeCharStartPredicate, name)", 
               "TfidfPredicate: (0.2, name)", 
               "SimplePredicate: (sameFiveCharStartPredicate, name)", 
               "TfidfPredicate: (0.6, name)", 
               "SimplePredicate: (firstTokenPredicate, name)", 
               "SimplePredicate: (wholeFieldPredicate, name)", 
               "TfidfPredicate: (0.8, name)", 
               "SimplePredicate: (commonFourGram, name)", 
               "SimplePredicate: (sameSevenCharStartPredicate, name)"])


    
class TfidfTest(unittest.TestCase):
  def setUp(self):
    self.data_d = {
                     100 : {"name": "Bob", "age": "50", "dataset": 0},
                     105 : {"name": "Charlie", "age": "75", "dataset": 1},
                     110 : {"name": "Meredith", "age": "40", "dataset": 1},
                     115 : {"name": "Sue", "age": "10", "dataset": 0},
                     120 : {"name": "Jimbo", "age": "21","dataset": 0},
                     125 : {"name": "Jimbo", "age": "21", "dataset": 0},
                     130 : {"name": "Willy", "age": "35", "dataset": 0},
                     135 : {"name": "Willy", "age": "35", "dataset": 1},
                     140 : {"name": "Martha", "age": "19", "dataset": 1},
                     145 : {"name": "Kyle", "age": "27", "dataset": 0},
                  }
    

  def test_unconstrained_inverted_index(self):

    blocker = dedupe.blocking.DedupeBlocker([dedupe.blocking.TfidfPredicate(0.0, "name")])

    blocker.tfIdfBlock(((record_id, record["name"]) 
                        for record_id, record 
                        in self.data_d.iteritems()),
                       "name")

    canopy = list(blocker.tfidf_fields['name'])[0].canopy

    assert canopy == {120: 120, 130: 130, 125: 120, 135: 130}

    blocks = defaultdict(set)
    
    for block_key, record_id in blocker(self.data_d.items()) :
      blocks[block_key].add(record_id)

    assert set([frozenset(block) for block in blocks.values()]) ==\
        set([frozenset([120, 125]), frozenset([130, 135])])

  def test_constrained_inverted_index(self):

    blocker = dedupe.blocking.RecordLinkBlocker([dedupe.blocking.TfidfPredicate(0.0, "name")])

    fields_1 = dict((record_id, record["name"]) 
                    for record_id, record 
                    in self.data_d.iteritems()
                    if record["dataset"] == 0)

    fields_2 = dict((record_id, record["name"]) 
                    for record_id, record 
                    in self.data_d.iteritems()
                    if record["dataset"] == 1)

    blocker.tfIdfIndex(fields_2.items(), "name")

    canopy = list(blocker.tfidf_fields['name'])[0].canopy

    blocks = defaultdict(set)
    
    for block_key, record_id in blocker(self.data_d.items()) :
      blocks[block_key].add(record_id)

    assert sorted(each for each in blocks.values() if len(each) >1) ==\
      [set((130, 135))]

    

if __name__ == "__main__":
    unittest.main()
