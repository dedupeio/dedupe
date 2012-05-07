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

  #v_matrix = minimum distance matrix
  v_matrix = [[None for _ in xrange(length1+1)] for _ in xrange(length2+1)]

  # define base case of recurrences
  # Matrix(0,0) = 0
  # Matrix(0,j) = gapWeight + spaceWeight * j
  v_matrix[0] = f_matrix = [(j * spaceWeight + gapWeight) for j in xrange(length1 + 1)]
  v_matrix[0][0] = 0

  #e_matrix: minimum distance matrix when string1 prefix is left aligned to
  #string2
  #e_matrix = [row[:] for row in v_matrix]

  #f_matrix: minimum distance matrix when string1 prefix is right
  #aligned to string2
  #f_matrix = [row[:] for row in v_matrix]

  for i in xrange(1,length2 + 1) :
    # Matrix(i,0) = gapWeight + spaceWeight * i
    v_matrix[i][0] = e = i * spaceWeight + gapWeight

    for j in xrange(1,length1 + 1) :
      if string2[i-1] == string1[j-1] :  
        g = v_matrix[i-1][j-1] + matchWeight
      else :
        g = v_matrix[i-1][j-1] + mismatchWeight

      e = (e + spaceWeight
           if e > v_matrix[i][j-1] + gapWeight
           else v_matrix[i][j-1] + gapWeight + spaceWeight)

      #print "g: ", g_matrix[row][col]
      #e_matrix[i][j] = (min(e_matrix[i][j-1],
      #                      v_matrix[i][j-1] + gapWeight)
      #                  + spaceWeight)
      #print "e: ", e_matrix[row][col]
      
      f_matrix[j] = (f_matrix[j] + spaceWeight
                     if f_matrix[j] > v_matrix[i][j-1] + gapWeight
                     else v_matrix[i-1][j] + gapWeight + spaceWeight)

      #print "f: ", f_matrix[row][col]
      v_matrix[i][j] = min(e,
                           f_matrix[j],
                           g)
      #print "v: ", v_matrix[row][col]
  
  #print "Affine Gap v_matrix:"
  #print_matrix(g_matrix)
  #print_matrix(e_matrix)
  #print_matrix(f_matrix)
  #print_matrix(v_matrix)
  
  return v_matrix[length2][length1]/float(length1 + length2)

if __name__ == "__main__" :
    import cProfile
    def test() :
        for i in xrange(100000) :
            string1 = 'asdf'
            string2 = 'fdsa'
            distance = affineGapDistance(string1, string2)		

    cProfile.run('test()')
