import cProfile
from affinegap import affineGapDistance

def performanceTest() :
  for i in xrange(10000) :
    string2 = 'asdfa;dsjnfas;dfasdsdfasdf asdf'
    string1 = 'fdsa576576576'
    distance = affineGapDistance(string1, string2)

def correctnessTest() :
  print affineGapDistance('a', 'b', -5, 5, 5, 1) == 5
  print affineGapDistance('ab', 'cd', -5, 5, 5, 1) == 10
  print affineGapDistance('ab', 'cde', -5, 5, 5, 1) == 15
  print affineGapDistance('a', 'cde', -5, 5, 5, 1) == 12
  print affineGapDistance('a', 'cd', -5, 5, 5, 1) == 11
  print affineGapDistance('b', 'a', -5, 5, 5, 1) == 5
  print affineGapDistance('a', 'a', -5, 5, 5, 1) == -5
  print affineGapDistance('a', '', -5, 5, 5, 1) == 6
  print affineGapDistance('aba', 'aaa', -5, 5, 5, 1) == -5
  print affineGapDistance('aaa', 'aba', -5, 5, 5, 1) == -5
  print affineGapDistance('aaa', 'aa', -5, 5, 5, 1) == -4
  print affineGapDistance('aaa', 'a', -5, 5, 5, 1) == 2
  print affineGapDistance('aaa', '', -5, 5, 5, 1) == 8
  print affineGapDistance('aaa', 'abba', -5, 5, 5, 1) == 1
  print affineGapDistance('0', '1', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '2', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '3', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '4', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '5', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '6', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '7', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '8', -5, 5, 5, 1) == 10
  print affineGapDistance('0', '9', -5, 5, 5, 1) == 10
  print affineGapDistance('0', 'a', -5, 5, 5, 1) == 5
  print affineGapDistance('0', '', -5, 5, 5, 1) == 6



        
correctnessTest()
#performanceTest()
