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

class Match(unittest.TestCase) :
  def test_initialize_fields(self) :
    matcher = dedupe.api.Matching()

    assert matcher.matches is None
    assert matcher.blocker is None



class ActiveMatch(unittest.TestCase) :
  def test_initialize_fields(self) :
    self.assertRaises(TypeError, dedupe.api.ActiveMatching)
    self.assertRaises(ValueError, dedupe.api.ActiveMatching, [])

    matcher = dedupe.api.ActiveMatching({},)

    assert matcher.matches is None
    assert matcher.blocker is None


  def test_add_training(self) :
    training_pairs = {'distinct' : DATA_SAMPLE[0:3],
                      'match' : DATA_SAMPLE[3:5]}
    matcher = dedupe.api.ActiveMatching({ 'name' : {'type': 'String'}, 
                                          'age'  : {'type': 'String'}})

    matcher._addTrainingData(training_pairs)
    numpy.testing.assert_equal(matcher.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match'])
    numpy.testing.assert_almost_equal(matcher.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]),
                                      4)

    matcher._addTrainingData(training_pairs)
    numpy.testing.assert_equal(matcher.training_data['label'],
                               ['distinct', 'distinct', 'distinct', 
                                'match', 'match']*2)

    numpy.testing.assert_almost_equal(matcher.training_data['distances'],
                                      numpy.array(
                                        [[ 5.5, 5.0178],
                                         [ 5.5, 3.4431],
                                         [ 5.5, 3.7750],
                                         [ 3.0, 5.125 ],
                                         [ 5.5, 4.8333]]*2),
                                      4)



class DedupeTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123) 
    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    self.deduper = dedupe.Dedupe(fields)

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

  def test_sample(self) :
    data_sample = self.deduper._sample(
      {'1' : {'name' : 'Frank', 'age' : '72'},
       '2' : {'name' : 'Bob', 'age' : '27'},
       '3' : {'name' : 'Jane', 'age' : '28'}}, 10)


    names = [(pair[0]['name'], pair[1]['name']) for pair in data_sample]
    assert set(names) == set([("Frank", "Bob"), 
                              ("Frank", "Jane"),
                              ("Jane", "Bob")])

    self.deduper.sample({'1' : {'name' : 'Frank', 'age' : '72'},
                         '2' : {'name' : 'Bob', 'age' : '27'},
                         '3' : {'name' : 'Jane', 'age' : '28'}}, 10)

    assert self.deduper.data_sample == data_sample




class LinkTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123) 
    fields =  { 'name' : {'type': 'String'}, 
                'age'  : {'type': 'String'},
              }
    self.linker = dedupe.RecordLink(fields)

  def test_blockPairs(self) :
    self.assertRaises(ValueError, self.linker.blockedPairs, ((),))
    self.assertRaises(ValueError, self.linker.blockedPairs, ({1:2},))
    self.assertRaises(ValueError, self.linker.blockedPairs, ({'name':'Frank', 'age':21},))
    self.assertRaises(ValueError, self.linker.blockedPairs, ({'1' : {'name' : 'Frank',
                                                                      'height' : 72}},))
    assert [] == list(self.linker.blockedPairs((({'1' : {'name' : 'Frank',
                                                         'age' : 72}},
                                                 {}),)))
    assert list(self.linker.blockedPairs((({'1' : {'name' : 'Frank',
                                                   'age' : 72}},
                                           {'2' : {'name' : 'Bob',
                                                   'age' : 27}}),))) == \
                  [(('1', {'age': 72, 'name': 'Frank'}), 
                    ('2', {'age': 27, 'name': 'Bob'}))]

  def test_sample(self) :
    data_sample = self.linker._sample(
      {'1' : {'name' : 'Frank', 'age' : '72'}},
      {'2' : {'name' : 'Bob', 'age' : '27'},
       '3' : {'name' : 'Jane', 'age' : '28'}}, 10)

    names = [(pair[0]['name'], pair[1]['name']) for pair in data_sample]
    assert set(names) == set([("Frank", "Bob"), ("Frank", "Jane")])

    self.linker.sample({'1' : {'name' : 'Frank', 'age' : '72'}},
                       {'2' : {'name' : 'Bob', 'age' : '27'},
                        '3' : {'name' : 'Jane', 'age' : '28'}}, 10)

    assert self.linker.data_sample == data_sample

      
      



if __name__ == "__main__":
    unittest.main()
