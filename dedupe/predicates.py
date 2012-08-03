import re

#returns the field as a tuple
def wholeFieldPredicate(field) :
  return (field, )
  
#returns the tokens in the field as a tuple, split on whitespace
def tokenFieldPredicate(field) :
  return tuple(field.split())

# Contain common integer
def commonIntegerPredicate(field) :
    return tuple(re.findall("\d+", field))

def nearIntegersPredicate(field) :
    ints = sorted([int(i) for i in re.findall("\d+", field)])
    return tuple([(i-1, i, i+1) for i in ints])


def commonFourGram(field) :
    return tuple([field[pos:pos + 4] for pos in xrange(0, len(field), 4)])

def commonSixGram(field) :
    return tuple([field[pos:pos + 6] for pos in xrange(0, len(field), 6)])

def sameThreeCharStartPredicate(field) :
    return (field[:3],)

def sameFiveCharStartPredicate(field) :
    return (field[:5],)

def sameSevenCharStartPredicate(field) :
    return (field[:7],)

if __name__ == '__main__':
  field = '123 16th st'
  print wholeFieldPredicate(field) == ('123 16th st',)
  print tokenFieldPredicate(field) == ['123', '16th', 'st']
  print commonIntegerPredicate(field) == ['123', '16']
  print sameThreeCharStartPredicate(field) == ('123',)
  print sameFiveCharStartPredicate(field) == ('123 1',)
  print sameSevenCharStartPredicate(field) == ('123 16t',)
  print nearIntegersPredicate(field) == [(15, 16, 17), (122, 123, 124)]
  print commonFourGram(field) == ['123 ', '16th', ' st']
  print commonSixGram(field) == ['123 16', 'th st']

