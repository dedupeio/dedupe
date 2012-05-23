def print_matrix(m):
  print ' '
  for line in m:
    spTupel = ()
    breite = len(line)
    for column in line:
      spTupel = spTupel + (column, )
      print "%3i"*breite % spTupel

# Calculate the affine gap distance between two strings 
#
# Default weights are from Alvaro Monge and Charles Elkan, 1996, 
# "The field matching problem: Algorithms and applications" 
# http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.23.9685


#@profile
def affineGapDistance(string1, string2,
                      matchWeight = -5,
                      mismatchWeight = 5,
                      gapWeight = 4,
                      spaceWeight = 1):

     
  if (string1 == string2 and
      matchWeight == min(matchWeight,
                         mismatchWeight,
                         gapWeight)):
      return matchWeight * len(string1)

  string1 = list(enumerate(string1,1))
  string2 = list(enumerate(string2,1))

  length1 = len(string1)
  length2 = len(string2)
    
  if length1 < length2 :
      string1, string2 = string2, string1
      length1, length2 = length2, length1

  #set up recurrence matrices
  #
  # Base conditions 
  # V(0,0) = F(0,0) = 0
  # V(0,j) = F(0,j) = gapWeight + spaceWeight * j
  v_current = [0] + range(gapWeight + spaceWeight,
                          gapWeight + spaceWeight * (length1 + 1),
                          spaceWeight)
  

  f = [1000000] * len(v_current)
  
  for i, char2 in string2 :
    v_previous = v_current[:]

    # Base conditions  
    # V(i,0) = E(i,0) = gapWeight + spaceWeight * i
    v_current[0] = i * spaceWeight + gapWeight
    e = 1000000000

    for j, char1 in string1 :

      # E: minimum distance matrix when string1 prefix is left aligned
      # to string2
      #
      # E(i,j) = min(E(i,j-1), V(i,j-1) + gapWeight) + spaceWeight

      e = (e + spaceWeight
           if e < v_current[j-1] + gapWeight
           else v_current[j-1] + gapWeight + spaceWeight)
      
      # F: minimum distance matrix when string1 prefix is right
      # aligned to string2
      #
      # F(i,j) = min(F(i-1,j), V(i-1,j) + gapWeight) + spaceWeight
      f_j = f[j]

      f[j] = f_j = (f_j + spaceWeight
                    if f_j < v_previous[j] + gapWeight
                    else v_previous[j] + gapWeight + spaceWeight)
              
      # G: minimum distance matrix when string1 prefix is aligned to
      # string2
      #
      # G(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)  
      g = (v_previous[j-1] + matchWeight
           if char2 == char1
           else v_previous[j-1] + mismatchWeight
           )


      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      if e < g  :
        if e < f_j :
          v_current[j] = e
        else :
          v_current[j] = f_j
      elif g < f_j :
        v_current[j] = g
      else :
        v_current[j] = f_j




  return v_current[length1]

def normalizedAffineGapDistance(string1, string2,
                      matchWeight = -5,
                      mismatchWeight = 5,
                      gapWeight = 5,
                      spaceWeight = 1) :

    normalizer = float(len(string1) + len(string2))
    alpha = max(matchWeight, misMatchWeight, gapWeight, spaceWeight)
    
    gld = affineGapDistance(string1, string2,
                            matchWeight,
                            mismatchWeight,
                            gapWeight,
                            spaceWeight)
    #return (alpha * normalizer - gld) / normalizer
    return ld / normalizer


if __name__ == "__main__" :
    import cProfile
    def performanceTest() :
        for i in xrange(10000) :
            string2 = 'asdfa;dsjnfas;dfasdsdfasdf asdf'
            string1 = 'fdsa576576576'
            distance = affineGapDistance(string1, string2)

    def correctnessTest() :
        print affineGapDistance('a', 'b', -5, 5, 5, 1) == 5
        print affineGapDistance('b', 'a', -5, 5, 5, 1) == 5
        print affineGapDistance('a', 'a', -5, 5, 5, 1) == -5
        print affineGapDistance('a', '', -5, 5, 5, 1) == 6
        print affineGapDistance('aba', 'aaa', -5, 5, 5, 1) == -5
        print affineGapDistance('aaa', 'aba', -5, 5, 5, 1) == -5
        print affineGapDistance('aaa', 'aa', -5, 5, 5, 1) == -4
        print affineGapDistance('aaa', 'a', -5, 5, 5, 1) == 2
        print affineGapDistance('aaa', '', -5, 5, 5, 1) == 8
        print affineGapDistance('aaa', 'abba', -5, 5, 5, 1) == 1
        print affineGapDistance('abba', 'aaa', -5, 5, 5, 1) == 1

        
    correctnessTest()
    performanceTest()
