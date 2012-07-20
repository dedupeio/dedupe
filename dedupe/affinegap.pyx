#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from cpython cimport array

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

  # array.array is part of Cython 0.17 http://bit.ly/LDxyj3 . We are
  # using the development branch now.
  cdef array.array D = array.array('f')
  array.resize(D, length1+1)
  array.zero(D)

  cdef array.array V_current = array.copy(D)
  cdef array.array V_previous = array.copy(V_current)

  cdef char char1, char2
  cdef int i, j
  cdef float e, g

  # Set up Recurrence relations
  #
  # Base conditions
  # V(0,0) = 0
  # V(0,j) = gapWeight + spaceWeight * i
  # D(0,j) = Infinity
  V_current._f[0] = 0
  for j in range(1, length1 + 1) :
    V_current._f[j] = gapWeight + spaceWeight * j
    D._f[j] = limits.INT_MAX

  for i in range(1, length2+1) :
    char2 = string2[i-1]
    # V_previous = V_current
    for _ in range(0, length1 + 1) :
        V_previous._f[_] = V_current._f[_]

    # Base conditions  
    # V(i,0) = gapWeight + spaceWeight * i
    # I(i,0) = Infinity 
    V_current._f[0] = gapWeight + spaceWeight * i
    I = limits.INT_MAX
  
    for j in range(1, length1 + 1) :
      char1 = string1[j-1]

      # I(i,j) is the edit distance if the jth character of string 1
      # was inserted into string 2.
      #
      # I(i,j) = min(I(i,j-1), V(i,j-1) + gapWeight) + spaceWeight

      if j <= length2 :
        I = min(I, V_current._f[j-1] + gapWeight) + spaceWeight
      else :
        # Pay less for abbreviations
        # i.e. 'spago (los angeles) to 'spago'
        I = (min(I, V_current._f[j-1] + gapWeight * abbreviation_scale)
             + spaceWeight * abbreviation_scale)
        
      # D(i,j) is the edit distance if the ith character of string 2
      # was deleted from string 1
      #
      # D(i,j) = min((i-1,j), V(i-1,j) + gapWeight) + spaceWeight
      D._f[j] = min(D._f[j], V_previous._f[j] + gapWeight) + spaceWeight
              
      # M(i,j) is the edit distance if the ith and jth characters
      # match or mismatch
      #
      # M(i,j) = V(i-1,j-1) + (matchWeight | misMatchWeight)  
      if char2 == char1 :
        M = V_previous._f[j-1] + matchWeight
      else:
        M = V_previous._f[j-1] + mismatchWeight
      
      # V(i,j) is the minimum edit distance 
      #  
      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      V_current._f[j] = min(I, D._f[j], M)

  return V_current._f[length1]

cpdef float normalizedAffineGapDistance(char *string1, char *string2,
                                        float matchWeight = -5,
                                        float mismatchWeight = 5,
                                        float gapWeight = 4,
                                        float spaceWeight = 1,
                                        float abbreviation_scale = .5) :
  
    cdef float normalizer = len(string1) + len(string2)

    if normalizer == 0 :
      return 0

    cdef float distance = affineGapDistance(string1, string2,
                                            matchWeight,
                                            mismatchWeight,
                                            gapWeight,
                                            spaceWeight,
                                            abbreviation_scale)

    return distance/normalizer


