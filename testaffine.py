import cProfile
from affinegap import affineGapDistance, normalizedAffineGapDistance

def performanceTest() :
  for i in xrange(100000) :
    string2 = 'asdfa;dsjnfas;dfasdsdfasdf asdf'
    string1 = 'fdsa576576576'
    distance = affineGapDistance(string1, string2)

def correctnessTest() :
  print 'Affine Gap Tests'
  print affineGapDistance('a', 'b', -5, 5, 5, 1) == 5
  print affineGapDistance('ab', 'cd', -5, 5, 5, 1) == 10
  print affineGapDistance('ab', 'cde', -5, 5, 5, 1) == 13
  print affineGapDistance('a', 'cde', -5, 5, 5, 1) == 8.5
  print affineGapDistance('a', 'cd', -5, 5, 5, 1) == 8
  print affineGapDistance('b', 'a', -5, 5, 5, 1) == 5
  print affineGapDistance('a', 'a', -5, 5, 5, 1) == -5
  print affineGapDistance('a', '', -5, 5, 5, 1) == 3
  print affineGapDistance('', '', -5, 5, 5, 1) == 0
  print affineGapDistance('aba', 'aaa', -5, 5, 5, 1) == -5
  print affineGapDistance('aaa', 'aba', -5, 5, 5, 1) == -5
  print affineGapDistance('aaa', 'aa', -5, 5, 5, 1) == -7
  print affineGapDistance('aaa', 'a', -5, 5, 5, 1) == -1.5
  print affineGapDistance('aaa', '', -5, 5, 5, 1) == 4
  print affineGapDistance('aaa', 'abba', -5, 5, 5, 1) == 1
  print 'Normalized Affine Gap Tests'
  print normalizedAffineGapDistance('', '', -5, 5, 5, 1) == 0


        
correctnessTest()
cProfile.run("performanceTest()")
