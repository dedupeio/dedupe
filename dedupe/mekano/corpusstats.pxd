cimport atomvector

cdef class CorpusStats:
    cdef public atomvector.AtomVector df
    cdef public int N
    
    cpdef add(self, atomvector.AtomVector av)
    cpdef int getDF(self, int a)
    cpdef int getN(self)
    
    
