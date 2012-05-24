from libc cimport limits 
#cython: boundscheck=False, wraparound=False
DEF ArraySize = 1000

# Calculate the affine gap distance between two strings 
#
# Default weights are from Alvaro Monge and Charles Elkan, 1996, 
# "The field matching problem: Algorithms and applications" 
# http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.23.9685

cpdef float affineGapDistance(char *string1, char *string2,
                              float matchWeight = -5,
                              float mismatchWeight = 5,
                              float gapWeight = 4,
                              float spaceWeight = 1,
                              float abbreviation_scale = .5):

  cdef int length1 = len(string1)

  if (string1 == string2 and
      matchWeight == min(matchWeight,
                         mismatchWeight,
                         gapWeight)):
      return matchWeight * length1

  cdef int length2 = len(string2)

  if length1 == 0 or length2 == 0 :
    return (gapWeight + spaceWeight * (length1 + length2)) * abbreviation_scale

  if length1 < length2 :
      string1, string2 = string2, string1
      length1, length2 = length2, length1

      
  # This is a terrible part of the code. Pythonic list indexing is
  # very slow, and doing it this way saves an enormous amount of
  # time. However it's pretty brittle
  #
  # Cython 0.17 looks like it will have fix for this:
  # http://bit.ly/LDxyj3

  #cdef float D[ArraySize]
  #cdef float V_current[ArraySize]
  #cdef float V_previous[ArraySize]

  # With blocking, this is acceptable. It is about 20 times slower
  # than using a a buffer.
  D = [0] * (length1+1) 
  V_current = [0] * (length1+1) 
  V_previous = [0] * (length1+1) 

  cdef char char1, char2
  cdef int i, j
  cdef float e, g

  # Set up Recurrence relations
  #
  # Base conditions
  # V(0,0) = 0
  # V(0,j) = gapWeight + spaceWeight * i
  # D(0,j) = Infinity
  V_current[0] = 0
  for j in range(1, length1 + 1) :
    V_current[j] = gapWeight + spaceWeight * j
    D[j] = limits.INT_MAX

  for i in range(1, length2+1) :
    char2 = string2[i-1]
    # v_previous = v_current, probably a better way to do this. This
    # will also be fixable in cython 0.17
    for _ in range(0, length1+1) :	
      V_previous[_] = V_current[_]

    # Base conditions  
    # V(i,0) = gapWeight + spaceWeight * i
    # I(i,0) = Infinity 
    V_current[0] = i * spaceWeight + gapWeight
    I = limits.INT_MAX
  
    for j in range(1, length1 + 1) :

      # Pay less for abbreviations
      # i.e. 'spago (los angeles) to 'spago'
      if j > length2 :
        I = (min(I, V_current[j-1] + gapWeight * abbreviation_scale)
             + spaceWeight * abbreviation_scale)
        V_current[j] = I
        continue
            
      char1 = string1[j-1]
      # I(i,j) is the edit distance if the jth character was inserted.
      #
      # I(i,j) = min(I(i,j-1), V(i,j-1) + gapWeight) + spaceWeight
      I = min(I, V_current[j-1] + gapWeight) + spaceWeight
      
      # D(i,j) is the edit distance if the ith character was deleted
      #
      # D(i,j) = min((i-1,j), V(i-1,j) + gapWeight) + spaceWeight
      D[j] = min(D[j], V_previous[j] + gapWeight) + spaceWeight
              
      # M(i,j) is the edit distance if the ith and jth characters
      # match or mismatch
      #
      # M(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)  
      if char2 == char1 :
        M = V_previous[j-1] + matchWeight
      else:
        M = V_previous[j-1] + mismatchWeight
      
      # V(i,j) is the minimum edit distance 
      #  
      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      V_current[j] = min(I, D[j], M)

  return V_current[length1]

cpdef float normalizedAffineGapDistance(char *string1, char *string2,
                                        float matchWeight = -5,
                                        float mismatchWeight = 5,
                                        float gapWeight = 4,
                                        float spaceWeight = 1) :
  
    cdef float normalizer = len(string1) + len(string2)
    cdef float alpha = gapWeight + spaceWeight
    
    cdef float distance = affineGapDistance(string1, string2,
                             matchWeight,
                             mismatchWeight,
                             gapWeight ,
                             spaceWeight)

    # Normalization proposed by Li Yujian and Li Bo's in "A Normalized
    # Levenshtein Distance Metric" http://dx.doi.org/10.1109/TPAMI.2007.1078
    return (2 * distance)/(alpha * normalizer + distance)


