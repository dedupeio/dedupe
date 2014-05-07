import dedupe
import unittest

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

    
class TfidfTest(unittest.TestCase):
  def setUp(self):
    self.field = "Hello World world"
    self.record_id = 20
    self.data_d = {
                     100 : {"name": "Bob", "age": "50", "dataset": 0},
                     105 : {"name": "Charlie", "age": "75", "dataset": 1},
                     110 : {"name": "Meredith", "age": "40", "dataset": 1},
                     115 : {"name": "Sue", "age": "10", "dataset": 0},
                     120 : {"name": "Jimbo", "age": "21","dataset": 1},
                     125 : {"name": "Jimbo", "age": "21", "dataset": 0},
                     130 : {"name": "Willy", "age": "35", "dataset": 0},
                     135 : {"name": "Willy", "age": "35", "dataset": 1},
                     140 : {"name": "Martha", "age": "19", "dataset": 1},
                     145 : {"name": "Kyle", "age": "27", "dataset": 0},
                  }
    
    self.tfidf_fields = ["name"]



  def test_unconstrained_inverted_index(self):

    blocker = dedupe.blocking.DedupeBlocker([dedupe.blocking.TfidfPredicate(0.0, "name")])

    blocker.tfIdfBlock(((record_id, record["name"]) 
                        for record_id, record 
                        in self.data_d.iteritems()),
                       "name")

    canopy = list(blocker.tfidf_fields['name'])[0].canopy

    assert canopy == {120: 120, 130: 130, 125: 120, 135: 130}

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

    blocker.tfIdfBlock(fields_1.items(), fields_2.items(), "name")

    canopy = list(blocker.tfidf_fields['name'])[0].canopy

    assert set(canopy.values()) <= set(fields_1.keys())

    assert canopy == {120: 125, 135: 130, 130: 130, 125: 125}




if __name__ == "__main__":
    unittest.main()
