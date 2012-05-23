from libc cimport limits 
#cython: boundscheck=False, wraparound=False
DEF ArraySize = 1000
#import numpy as np
#cimport numpy as np
#DTYPE = np.int
#ctypedef np.int_t DTYPE_t

# Calculate the affine gap distance between two strings 
#
# Default weights are from Alvaro Monge and Charles Elkan, 1996, 
# "The field matching problem: Algorithms and applications" 
# http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.23.9685

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

      
  # This is a terrible part of the code. Pythonic list indexing is
  # very slow, and doing it this way saves an enormous amount of
  # time. However it's pretty brittle
  #
  # Cython 0.17 looks like it will have fix for this:
  # http://bit.ly/LDxyj3

  cdef int f[ArraySize]
  cdef int v_current[ArraySize]
  cdef int v_previous[ArraySize]

  # This is less brittle, but requires that the end user have numpy
  # installed
  #cdef np.ndarray[DTYPE_t] f = np.zeros(length1 + 1, dtype=DTYPE)
  #cdef np.ndarray[DTYPE_t] v_current = np.zeros(length1 + 1, dtype=DTYPE)
  #cdef np.ndarray[DTYPE_t] v_previous = np.zeros(length1 + 1, dtype=DTYPE)

  cdef char char1, char2
  cdef int i, j, e, g

  # Set up Recurrence relations
  #
  # Base conditions
  # V(0,0) = 0
  # V(0,j) = gapWeight + spaceWeight * i
  # F(0,j) = Infinity
  v_current[0] = 0
  for i in range(1, length1 + 1) :
    v_current[i] = gapWeight + spaceWeight * i
    f[i] = limits.INT_MAX

  for i in range(1, length2+1) :
    char2 = string2[i-1]
    # v_previous = v_current, probably a better way to do this. This
    # will also be fixable in cython 0.17
    for _ in range(0, length1+1) :	
      v_previous[_] = v_current[_]

    # Base conditions  
    # V(i,0) = gapWeight + spaceWeight * i
    # E(i,0) = Infinity 
    v_current[0] = i * spaceWeight + gapWeight
    e = limits.INT_MAX
  
    for j in range(1, length1 + 1) :
      char1 = string1[j-1]
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
      # if the two characters are different integers, then pay double
      # the normal cost for mismatch
      #elif 47 < char2 < 59 and 47 < char1 < 59 :
      #  g = v_previous[j-1] + mismatchWeight * 2
      else:
        g = v_previous[j-1] + mismatchWeight
      


      # V(i,j) = min(E(i,j), F(i,j), G(i,j))
      v_current[j] = min(e, g, f[j])




  return v_current[j]

def normalizedAffineGapDistance(char *string1, char *string2,
                      int matchWeight = -5,
                      int mismatchWeight = 5,
                      int gapWeight = 5,
                      int spaceWeight = 1) :

    cdef float normalizer = float(len(string1) + len(string2))
    cdef float alpha = max(matchWeight, mismatchWeight, gapWeight, spaceWeight)
    
    cdef float distance = affineGapDistance(string1, string2,
                             matchWeight,
                             mismatchWeight,
                             gapWeight ,
                             spaceWeight)

    #return (alpha * normalizer - distance)/2
    return distance/normalizer

