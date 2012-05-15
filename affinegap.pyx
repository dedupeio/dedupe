#cython: boundscheck=False, wraparound=False
DEF ArraySize = 1000

#calculate the affine gap distance between 2 strings default weights
#taken from page 28 of Bilenko's Ph. D dissertation: Learnable
#Similarity Functions and their Application to Record Linkage and
#Clustering

cpdef affineGapDistance(char *string1, char *string2,
                      int matchWeight = -5,
                      int mismatchWeight = 5,
                      int gapWeight = 5,
                      int spaceWeight = 1):

  cdef int length1 = len(string1)
     
  if (string1 == string2 and
      matchWeight == min(matchWeight,
                         mismatchWeight,
                         gapWeight)):
      return matchWeight * length1

  cdef int length2 = len(string2)

  if length1 < length2 :
      string1, string2 = string2, string1
      length1, length2 = length2, length1

      
  #set up recurrence matrices
  #
  # Base conditions 
  # V(0,0) = F(0,0) = 0
  # V(0,j) = F(0,j) = gapWeight + spaceWeight * j

  # This is a terrible part of the code. Pythonic list indexing is
  # very slow, and doing it this way saves an enormous amount of
  # time. However it's pretty brittle
  cdef int f[ArraySize]
  cdef int v_current[ArraySize]
  cdef int v_previous[ArraySize]

  cdef char char1, char2
  cdef int i, j, e, g

  # Base conditions
  # V(0,0) = 0
  # V(0,j) = F(0,j) = gapWeight + spaceWeight * i
  v_current[0] = v_previous[0] = 0
  for i in range(1, length1 + 1) :
    f[i] = v_current[i] = v_previous[i] = gapWeight + spaceWeight * i

  for i in range(1, length2+1) :
    char2 = string2[i]

    # v_previous = v_current, probably a better way to do this
    for i in range(0, length1) :	
      v_previous[i] = v_current[i]

    # Base conditions  
    # V(i,0) = E(i,0) = gapWeight + spaceWeight * i
    v_current[0] = e = i * spaceWeight + gapWeight
  

    for j in range(1, length1 + 1) :
      char1 = string1[j]

      # E: minimum distance matrix when string1 prefix is left aligned
      # to string2
      #
      # E(i,j) = min(E(i,j-1), V(i,j-1) + gapWeight) + spaceWeight

      e = min(e, v_current[j-1] + gapWeight) + spaceWeight
      
      # F: minimum distance matrix when string1 prefix is right
      # aligned to string2
      #
      # F(i,j) = min(F(i-1,j), V(i-1,j) + gapWeight) + spaceWeight
      f[j] = min(f[j], v_previous[j] + gapWeight) + spaceWeight
              
      # G: minimum distance matrix when string1 prefix is aligned to
      # string2
      #
      # G(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)  
      if char2 == char1 :
        g = v_previous[j-1] + matchWeight
      else :
        g = v_previous[j-1] + mismatchWeight

      



      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      v_current[j] = min(e, g, f[j])




  return v_current[length1]

def normalizedAffineGapDistance(char *string1, char *string2,
                      int matchWeight = -5,
                      int mismatchWeight = 5,
                      int gapWeight = 5,
                      int spaceWeight = 1) :

    cdef float normalizer = float(len(string1) + len(string2))
    
    cdef float distance = affineGapDistance(string1, string2,
                             matchWeight,
                             mismatchWeight,
                             gapWeight ,
                             spaceWeight)

    return distance/normalizer


