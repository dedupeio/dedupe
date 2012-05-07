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

def affineGapDistance(string1, string2,
                      matchWeight = -5,
                      mismatchWeight = 5,
                      gapWeight = 5,
                      spaceWeight = 1):
  
  length1 = len(string1)
  length2 = len(string2)
  

  #set up recurrence matrices

  #V_matrix = minimum distance matrix
  v_matrix = [[None for _ in xrange(length1+1)] for _ in xrange(length2+1)]

  # define base case of recurrences
  # V(0,0) = F(0,0) = 0
  # V(0,j) = F(0,j) = gapWeight + spaceWeight * j
  
  v_matrix[0] = f = [(j * spaceWeight + gapWeight)
                     for j in xrange(length1 + 1)]
  v_matrix[0][0] = 0


  for i in xrange(1,length2 + 1) :
    # Base conditions  
    # V(i,0) = E(i,0) = gapWeight + spaceWeight * i
    v_matrix[i][0] = e = i * spaceWeight + gapWeight

    for j in xrange(1,length1 + 1) :

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
      if string2[i-1] == string1[j-1] :  
        g = v_matrix[i-1][j-1] + matchWeight
      else :
        g = v_matrix[i-1][j-1] + mismatchWeight

      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      v_matrix[i][j] = (e
                        if e < f[j]
                        else f[j]
                        if f[j] < g
                        else g)

  return v_matrix[length2][length1]/float(length1 + length2)

if __name__ == "__main__" :
    import cProfile
    def test() :
        for i in xrange(100000) :
            string1 = 'asdf'
            string2 = 'fdsa'
            distance = affineGapDistance(string1, string2)		

    cProfile.run('test()')
