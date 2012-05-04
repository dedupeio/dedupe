def print_matrix(m):
    print ' '
    for line in m:
        spTupel = ()
        breite = len(line)
        for column in line:
            spTupel = spTupel + (column, )
        print "%3i"*breite % spTupel

#string1 = raw_input('first word: ')
#string2 = raw_input('second word: ')

def defineCharacterScore(matchWeight, mismatchWeight) :
  def characterScore (char1, char2) :
    if (char1 == char2) :
      return matchWeight
    else :
      return mismatchWeight
  return characterScore
  
#calculate the affine gap distance between 2 strings
#default weights taken from page 28 of Bilenko's Ph. D dissertation: Learnable Similarity Functions and their Application to Record Linkage and Clustering
def affineGapDistance(string1, string2, matchWeight = -5, mismatchWeight = 5, gapWeight = 5, spaceWeight = 1):
  
  length1 = len(string1)
  length2 = len(string2)
  
  #set up scoring function with match and mismatch weights
  score = defineCharacterScore(matchWeight, mismatchWeight)
  
  #v_matrix = distance
  #e_matrix = distance when string1 prefix is left aligned to string2 
  #f_matrix = distance when string1 prefix is right aligned to string2
  #g_matrix = distence when string1 prefix and string2 prefix are aligned

  #define base case of recurrences
  v_matrix = [[(i * spaceWeight + gapWeight) for i in range(length1 + 1)]] * (length2 + 1)
  for row in range(length2 + 1):
    v_matrix[row] = [(i * spaceWeight + gapWeight) for i in range(row, row + length1 + 1)]
  
  #set up recurrence matrices
  e_matrix = [x[:] for x in v_matrix]
  f_matrix = [x[:] for x in v_matrix]
  g_matrix = [x[:] for x in v_matrix]
  
  for row in range(1,length2 + 1) :
    for col in range(1,length1 + 1) :
      g_matrix[row][col] = v_matrix[row-1][col-1] + score(string1[col-1],string2[row-1])
      #print "g: ", g_matrix[row][col]
      e_matrix[row][col] = min(e_matrix[row][col-1], v_matrix[row][col-1] + gapWeight) + spaceWeight
      #print "e: ", e_matrix[row][col]
      f_matrix[row][col] = min(f_matrix[row-1][col], v_matrix[row-1][col] + gapWeight) + spaceWeight
      #print "f: ", f_matrix[row][col]
      v_matrix[row][col] = min(e_matrix[row][col], f_matrix[row][col], g_matrix[row][col])
      #print "v: ", v_matrix[row][col]
  
  #print "Affine Gap v_matrix:"
  
  #print_matrix(g_matrix)
  #print_matrix(e_matrix)
  #print_matrix(f_matrix)
  #print_matrix(v_matrix)
  
  return v_matrix[length2][length1]
    
#distance = affineGapDistance(string1, string2)		
#print 'The Affine Gap Distance of ',string1, ' and ', string2, ' is ', distance