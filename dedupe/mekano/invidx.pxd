cimport dedupe.mekano.atomvector as atomvector
cimport dedupe.mekano.atomvectorstore as atomvectorstore
from dedupe.mekano.corpusstats cimport CorpusStats

cdef extern from "CUtils.h":

    ctypedef struct voidptr "void*":
        pass

    ctypedef struct a2avsitr "AtomToAtomVectorStoreMap::iterator":
        int first "operator->()->first"
        void* second "operator->()->second"
        void advance "operator++" ()
        int eq "operator==" (a2avsitr o)
        int neq "operator!=" (a2avsitr o)

    ctypedef struct a2avs "AtomToAtomVectorStoreMap":
        int size()
        void* ele "operator[]" (int a)
        a2avsitr find(int a)
        a2avsitr begin()
        a2avsitr end()


    a2avs* new_a2avs "new AtomToAtomVectorStoreMap" ()
    #void del_a2avs "delete" (a2avs* h)
    int has_a2avs(a2avs* dict, int a)
    void set_a2avs(a2avs* dict, int a, void* v)
    void del_a2avs(a2avs* dict)

cdef class InvertedIndex(CorpusStats):
    cdef a2avs* ii

    cpdef add(self, atomvector.AtomVector vec)
    cpdef int getDF(self, int a)
    cpdef int getN(self)
    cpdef atomvectorstore.AtomVectorStore getii(self, int a)
