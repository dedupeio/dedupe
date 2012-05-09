def print_matrix(m):
  print ' '
  for line in m:
    spTupel = ()
    breite = len(line)
    for column in line:
      spTupel = spTupel + (column, )
      print "%3i"*breite % spTupel

#calculate the affine gap distance between 2 strings default weights
#taken from page 28 of Bilenko's Ph. D dissertation: Learnable
#Similarity Functions and their Application to Record Linkage and
#Clustering

#@profile
def affineGapDistance(string1, string2,
                      matchWeight = -5,
                      mismatchWeight = 5,
                      gapWeight = 5,
                      spaceWeight = 1):

  string1 = list(enumerate(string1,1))
  string2 = list(enumerate(string2,1))

  length1 = len(string1)
  length2 = len(string2)
    
  if length1 < length2 :
      string1, string2 = string2, string1
      length1, length2 = length2, length1

  #set up recurrence matrices
  #
  #V_matrix = minimum distance matrix
  v_matrix = [[None] * (length1 + 1)
              for _ in string2]

  # Base conditions 
  # V(0,0) = F(0,0) = 0
  # V(0,j) = F(0,j) = gapWeight + spaceWeight * j

  f = [0] + [j * spaceWeight + gapWeight
             for j, char in string1]
  v_matrix.insert(0, f[:])
  
  
  for i, char2 in string2 :
    # Base conditions  
    # V(i,0) = E(i,0) = gapWeight + spaceWeight * i
    v_matrix[i][0] = e = i * spaceWeight + gapWeight
      
    for j, char1 in string1 :
      
      # E: minimum distance matrix when string1 prefix is left aligned
      # to string2
      #
      # E(i,j) = min(E(i,j-1), V(i,j-1) + gapWeight) + spaceWeight
      e = (e + spaceWeight
           if e < v_matrix[i][j-1] + gapWeight
           else v_matrix[i][j-1] + gapWeight + spaceWeight)
      
      # F: minimum distance matrix when string1 prefix is right
      # aligned to string2
      #
      # F(i,j) = min(F(i-1,j), V(i-1,j) + gapWeight) + spaceWeight
      f[j] = (f[j] + spaceWeight
              if f[j] < v_matrix[i-1][j] + gapWeight
              else v_matrix[i-1][j] + gapWeight + spaceWeight)
              
      # G: minimum distance matrix when string1 prefix is aligned to
      # string2
      #
      # G(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)  
      if char2 == char1 :  
        g = v_matrix[i-1][j-1] + matchWeight
      else :
        g = v_matrix[i-1][j-1] + mismatchWeight

      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      if e < g  :
        if e < f[j] :
          v_matrix[i][j] = e
        else :
          v_matrix[i][j] = f[j]
      elif g < f[j] :
        v_matrix[i][j] = g
      else :
        v_matrix[i][j] = f[j]

  return v_matrix[length2][length1]

def normalizedAffineGapDistance(string1, string2,
                      matchWeight = -5,
                      mismatchWeight = 5,
                      gapWeight = 5,
                      spaceWeight = 1) :

    normalizer = float(len(string1) + len(string2))
    
    return affineGapDistance(string1, string2,
                             matchWeight = -5,
                             mismatchWeight = 5,
                             gapWeight = 5,
                             spaceWeight = 1)/normalizer


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
