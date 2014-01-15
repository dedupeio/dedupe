import dedupe
import unittest
import random
import numpy

DATA = {  100 : {"name": "Bob", "age": "50"},
          105 : {"name": "Charlie", "age": "75"},
          110 : {"name": "Meredith", "age": "40"},
          115 : {"name": "Sue", "age": "10"}, 
          120 : {"name": "Jimmy", "age": "20"},
          125 : {"name": "Jimbo", "age": "21"},
          130 : {"name": "Willy", "age": "35"},
          135 : {"name": "William", "age": "35"},
          140 : {"name": "Martha", "age": "19"},
          145 : {"name": "Kyle", "age": "27"}
        }


class ConvenienceTest(unittest.TestCase):
  def test_data_sample(self):
    random.seed(123)
    numpy.random.seed(123)
    assert dedupe.dataSample(DATA ,5) == \
      (({'age': '27', 'name': 'Kyle'}, 
        {'age': '50', 'name': 'Bob'}), 
       ({'age': '50', 'name': 'Bob'}, 
        {'age': '21', 'name': 'Jimbo'}), 
       ({'age': '35', 'name': 'William'}, 
        {'age': '40', 'name': 'Meredith'}), 
       ({'age': '20', 'name': 'Jimmy'}, 
        {'age': '40', 'name': 'Meredith'}), 
       ({'age': '10', 'name': 'Sue'}, 
        {'age': '50', 'name': 'Bob'}))


if __name__ == "__main__":
    unittest.main()
