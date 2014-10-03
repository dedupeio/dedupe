import dedupe
import unittest
import random
import numpy
import warnings

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

data_dict = {    '1' : {'name' : 'Bob',         'age' : '51'},
                 '2' : {'name' : 'Linda',       'age' : '50'},
                 '3' : {'name' : 'Gene',        'age' : '12'},
                 '4' : {'name' : 'Tina',        'age' : '15'},
                 '5' : {'name' : 'Bob B.',      'age' : '51'},
                 '6' : {'name' : 'bob belcher', 'age' : '51'},
                 '7' : {'name' : 'linda ',      'age' : '50'} }

data_dict_2 = {  '1' : {'name' : 'BOB',         'age' : '51'},
                 '2' : {'name' : 'LINDA',       'age' : '50'},
                 '3' : {'name' : 'GENE',        'age' : '12'},
                 '4' : {'name' : 'TINA',        'age' : '15'},
                 '5' : {'name' : 'BOB B.',      'age' : '51'},
                 '6' : {'name' : 'BOB BELCHER', 'age' : '51'},
                 '7' : {'name' : 'LINDA ',      'age' : '50'} }

class Match(unittest.TestCase) :
  def test_initialize_fields(self) :
    matcher = dedupe.api.Matching()

    assert matcher.blocker is None



class ActiveMatch(unittest.TestCase) :
  def setUp(self) :
    self.field_definition = [{'field' : 'name', 'type': 'String'}, 
                             {'field' :'age', 'type': 'String'}]


  def test_initialize_fields(self) :
    self.assertRaises(TypeError, dedupe.api.ActiveMatching)

    matcher = dedupe.api.ActiveMatching({},)

    assert matcher.blocker is None


  def test_check_record(self) :
    matcher = dedupe.api.ActiveMatching(self.field_definition)

    self.assertRaises(ValueError, matcher._checkRecordPairType, ())
    self.assertRaises(ValueError, matcher._checkRecordPairType, (1,2))
    self.assertRaises(ValueError, matcher._checkRecordPairType, (1,2,3))
    self.assertRaises(ValueError, matcher._checkRecordPairType, ({},{}))

    matcher._checkRecordPairType(({'name' : 'Frank', 'age' : '72'},
                                  {'name' : 'Bob', 'age' : '27'}))


  def test_check_sample(self) :
    matcher = dedupe.api.ActiveMatching(self.field_definition)

    self.assertRaises(ValueError, 
                      matcher._checkDataSample, (i for i in range(10)))

    self.assertRaises(ValueError, 
                      matcher._checkDataSample, ((1, 2),))



    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      matcher._checkDataSample([])
      assert len(w) == 1
      assert str(w[-1].message) == "You submitted an empty data_sample"

  def test_add_training(self) :
    training_pairs = {'distinct' : DATA_SAMPLE[0:3],
                      'match' : DATA_SAMPLE[3:5]}
    matcher = dedupe.api.ActiveMatching(self.field_definition)

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

  def test_markPair(self) :
    good_training_pairs = {'distinct' : DATA_SAMPLE[0:3],
                      'match' : DATA_SAMPLE[3:5]}
    bad_training_pairs = {'non_dupes' : DATA_SAMPLE[0:3],
                      'match' : DATA_SAMPLE[3:5]}

    matcher = dedupe.api.ActiveMatching(self.field_definition)

    self.assertRaises(ValueError, matcher.markPairs, bad_training_pairs)

    matcher.markPairs(good_training_pairs)

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

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      matcher.markPairs({'match' : [], 'distinct' : []})
      assert len(w) == 1
      assert str(w[-1].message) == "Didn't return any labeled record pairs"

    


class DedupeTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123) 
    numpy.random.seed(456)

    field_definition = [{'field' : 'name', 'type': 'String'}, 
                        {'field' :'age', 'type': 'String'}]

    self.deduper = dedupe.Dedupe(field_definition)

  def test_blockPairs(self) :
    self.assertRaises(ValueError, self.deduper._blockedPairs, ({1:2},))
    self.assertRaises(ValueError, self.deduper._blockedPairs, ({'name':'Frank', 'age':21},))
    self.assertRaises(ValueError, self.deduper._blockedPairs, ({'1' : {'name' : 'Frank',
                                                                      'height' : 72}},))
    assert [] == list(self.deduper._blockedPairs(([('1', 
                                                    {'name' : 'Frank',
                                                     'age' : 72}, 
                                                    set([]))],)))
    assert list(self.deduper._blockedPairs(([('1', 
                                              {'name' : 'Frank',
                                               'age' : 72},
                                              set([])),
                                             ('2',
                                              {'name' : 'Bob',
                                               'age' : 27},
                                              set([]))],))) == \
                  [(('1', {'age': 72, 'name': 'Frank'}, set([])), 
                    ('2', {'age': 27, 'name': 'Bob'}, set([])))]

  def test_sample(self) :
    data_sample = self.deduper._randomSample(
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

  def test_randomSample(self) : 

    self.deduper.sample(data_dict, 5, 1)

    correct_result = (  ({'age': '51', 'name': 'Bob'}, {'age': '51', 'name': 'bob belcher'}),
                        ({'age': '50', 'name': 'Linda'}, {'age': '51', 'name': 'Bob B.'}),
                        ({'age': '12', 'name': 'Gene'}, {'age': '51', 'name': 'bob belcher'}),
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '15', 'name': 'Tina'}),
                        ({'age': '50', 'name': 'Linda'}, {'age': '51', 'name': 'bob belcher'})
    )

    assert self.deduper.data_sample == correct_result

  def test_blockedSample(self) :

    self.deduper.sample(data_dict, 5, 0)

    correct_result = (  ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'linda '}),
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'Bob'}),
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'linda '}),
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'Bob'}),
                        ({'age': '51', 'name': 'Bob'}, {'age': '51', 'name': 'bob belcher'})
                      )

    assert self.deduper.data_sample == correct_result

  def test_mixedSample(self) :

    self.deduper.sample(data_dict, 6, .5)

    correct_result = (  ({'age': '51', 'name': 'Bob'}, {'age': '51', 'name': 'bob belcher'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '51', 'name': 'Bob B.'}), 
                        ({'age': '12', 'name': 'Gene'}, {'age': '51', 'name': 'bob belcher'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'linda '}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'Bob'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'linda '})
                      )

    assert self.deduper.data_sample == correct_result


class LinkTest(unittest.TestCase):
  def setUp(self) : 
    random.seed(123)
    numpy.random.seed(456)

    field_definition = [{'field' : 'name', 'type': 'String'}, 
                        {'field' :'age', 'type': 'String'}]
    self.linker = dedupe.RecordLink(field_definition)

  def test_blockPairs(self) :
    self.assertRaises(ValueError, self.linker._blockedPairs, ({1:2},))
    self.assertRaises(ValueError, self.linker._blockedPairs, ({'name':'Frank', 'age':21},))
    self.assertRaises(ValueError, self.linker._blockedPairs, ({'1' : {'name' : 'Frank',
                                                                      'height' : 72}},))
    assert [] == list(self.linker._blockedPairs((([('1', 
                                                    {'name' : 'Frank',
                                                     'age' : 72}, 
                                                    set([]))],
                                                  []),)))
    assert list(self.linker._blockedPairs((([('1', {'name' : 'Frank',
                                                    'age' : 72}, set([]))],
                                            [('2', {'name' : 'Bob',
                                                    'age' : 27}, set([]))]),))) == \
                  [(('1', {'age': 72, 'name': 'Frank'}, set([])), 
                    ('2', {'age': 27, 'name': 'Bob'}, set([])))]

  def test_sample(self) :
    data_sample = self.linker._randomSample(
      {'1' : {'name' : 'Frank', 'age' : '72'}},
      {'2' : {'name' : 'Bob', 'age' : '27'},
       '3' : {'name' : 'Jane', 'age' : '28'}}, 10)

    names = [(pair[0]['name'], pair[1]['name']) for pair in data_sample]
    assert set(names) == set([("Frank", "Bob"), ("Frank", "Jane")])

    self.linker.sample({'1' : {'name' : 'Frank', 'age' : '72'}},
                       {'2' : {'name' : 'Bob', 'age' : '27'},
                        '3' : {'name' : 'Jane', 'age' : '28'}}, 10)

    assert self.linker.data_sample == data_sample

  def test_randomSample(self) :

    self.linker.sample( data_dict, data_dict_2, 5, 1)

    correct_result = (  ({'age': '50', 'name': 'linda '}, {'age': '51', 'name': 'BOB BELCHER'}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '50', 'name': 'LINDA'}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '15', 'name': 'TINA'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'LINDA '}), 
                        ({'age': '12', 'name': 'Gene'}, {'age': '15', 'name': 'TINA'})
                      )

    assert self.linker.data_sample == correct_result

  def test_blockedSample(self) :

    self.linker.sample( data_dict, data_dict_2, 5, 0)

    correct_result = (  ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'LINDA'}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'BOB B.'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'LINDA '}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'BOB B.'}), 
                        ({'age': '51', 'name': 'Bob'}, {'age': '51', 'name': 'BOB B.'})
                      )

    assert self.linker.data_sample == correct_result

  def test_mixedSample(self) :

    self.linker.sample( data_dict, data_dict_2, 6, .5)

    correct_result = (  ({'age': '51', 'name': 'Bob B.'}, {'age': '15', 'name': 'TINA'}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '12', 'name': 'GENE'}), 
                        ({'age': '50', 'name': 'linda '}, {'age': '50', 'name': 'LINDA'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'LINDA'}), 
                        ({'age': '51', 'name': 'Bob B.'}, {'age': '51', 'name': 'BOB B.'}), 
                        ({'age': '50', 'name': 'Linda'}, {'age': '50', 'name': 'LINDA '})
                      )

    assert self.linker.data_sample == correct_result

      
      



if __name__ == "__main__":
    unittest.main()
