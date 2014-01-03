cimport atomvector

cdef extern from "CUtils.h":

    # the hash map that internally stores the AtomVectorStore
    ctypedef struct storetype "CAVArray":
        int size()
        void* ele "operator[]" (int i)

    storetype* new_store "new CAVArray" ()
    void del_store(storetype* s)
    void add_store(storetype* s, void *av)

cdef class AtomVectorStore:
    cdef storetype *mystore
    cdef public int N
    cdef int itr

    cpdef add(AtomVectorStore self, atomvector.AtomVector av)
    cpdef atomvector.AtomVector getAt(self, int i)
