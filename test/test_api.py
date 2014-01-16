import dedupe
import unittest
import random
import numpy

DATA_SAMPLE = ((dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '50', 'name': 'Bob'})),
               (dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '35', 'name': 'William'})),
               (dedupe.core.frozendict({'age': '10', 'name': 'Sue'}), 
                dedupe.core.frozendict({'age': '35', 'name': 'William'})),
               (dedupe.core.frozendict({'age': '27', 'name': 'Kyle'}), 
                dedupe.core.frozendict({'age': '20', 'name': 'Jimmy'})),
               (dedupe.core.frozendict({'age': '75', 'name': 'Charlie'}), 
                dedupe.core.frozendict({'age': '21', 'name': 'Jimbo'})))



class DedupeInitializeTest(unittest.TestCase) :
  def test_initialize_fields(self) :
    self.assertRaises(TypeError, dedupe.Dedupe)
    self.assertRaises(ValueError, dedupe.Dedupe, [])

    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
               }
    deduper = dedupe.Dedupe(fields, [])

    assert deduper.matches is None
    assert deduper.blocker is None


  def test_base_predicates(self) :
    deduper = dedupe.Dedupe({'name' : {'type' : 'String'}}, [])
    string_predicates = (dedupe.predicates.wholeFieldPredicate,
                         dedupe.predicates.tokenFieldPredicate,
                         dedupe.predicates.commonIntegerPredicate,
                         dedupe.predicates.sameThreeCharStartPredicate,
                         dedupe.predicates.sameFiveCharStartPredicate,
                         dedupe.predicates.sameSevenCharStartPredicate,
                         dedupe.predicates.nearIntegersPredicate,
                         dedupe.predicates.commonFourGram,
                         dedupe.predicates.commonSixGram)

    tfidf_string_predicates = tuple([dedupe.tfidf.TfidfPredicate(threshold)
                                     for threshold
                                     in [0.2, 0.4, 0.6, 0.8]])

    assert deduper.blockerTypes() == {'String' : string_predicates + tfidf_string_predicates}


class DedupeClassTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123) 
    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    data_sample = DATA_SAMPLE
    self.deduper = dedupe.Dedupe(fields, data_sample)

  def test_blockPairs(self) :
    self.assertRaises(ValueError, self.deduper.blockedPairs, ((),))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({1:2},))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({'name':'Frank', 'age':21},))
    self.assertRaises(ValueError, self.deduper.blockedPairs, ({'1' : {'name' : 'Frank',
                                                                      'height' : 72}},))
    assert [] == list(self.deduper.blockedPairs(({'1' : {'name' : 'Frank',
                                                         'age' : 72}},)))
    assert list(self.deduper.blockedPairs(({'1' : {'name' : 'Frank',
                                                   'age' : 72},
                                            '2' : {'name' : 'Bob',
                                                   'age' : 27}},))) == \
                  [(('1', {'age': 72, 'name': 'Frank'}), 
                    ('2', {'age': 27, 'name': 'Bob'}))]

                                    

  def test_add_training(self) :
    training_pairs = {'distinct' : self.deduper.data_sample[0:3],
                      'match' : self.deduper.data_sample[3:5]}
    self.deduper._addTrainingData(training_pairs)
    numpy.testing.assert_equal(self.deduper.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match'])
    numpy.testing.assert_almost_equal(self.deduper.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]),
                                      4)
    self.deduper._addTrainingData(training_pairs)
    numpy.testing.assert_equal(self.deduper.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match']*2)

    numpy.testing.assert_almost_equal(self.deduper.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]*2),
                                      4)

if __name__ == "__main__":
    unittest.main()

