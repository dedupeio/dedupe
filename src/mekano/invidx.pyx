cimport dedupe.mekano.atomvector as atomvector
cimport dedupe.mekano.atomvectorstore as atomvectorstore
from dedupe.mekano.corpusstats cimport CorpusStats
import dedupe.mekano.atomvectorstore as atomvectorstore
from dedupe.mekano.corpusstats import CorpusStats
import os

cdef class InvertedIndex(CorpusStats):
    def __cinit__(self):
        self.ii = new_a2avs();

    def __init__(self):
        CorpusStats.__init__(self)
        self.N = 0

    def __dealloc__(self):
        del_a2avs(self.ii)

    def __reduce__(self):
        return InvertedIndex, (), self.__getstate__(), None, None

    def __getstate__(self):
        cdef a2avsitr it = self.ii.begin()
        cdef a2avsitr end = self.ii.end()
        cdef atomvectorstore.AtomVectorStore avs
        mymap = {}

        while (it.neq(end)):
            mymap[it.first] = <atomvectorstore.AtomVectorStore> it.second
            it.advance()

        return (self.N, mymap)

    def __setstate__(self, s):
        cdef int atom
        cdef atomvectorstore.AtomVectorStore avs
        
        self.N = s[0]

        for k, v in s[1].iteritems():
            atom = k
            avs = <atomvectorstore.AtomVectorStore> v

            set_a2avs(self.ii, atom, <void*>avs)

    cpdef add(self, atomvector.AtomVector vec):
        """
        add(vec, 1)
        The elements of AtomVector-like object 'vec' will be indexed,
        but the object that will appear under the invind of each
        such item is 'store'

        'store' can be id's of documents, or it could be the reference
        to the parent of 'vec' that is desirable to be retrieved.
        """
        cdef int a
        cdef double v
        cdef atomvector.dictitr itr, end
        cdef atomvectorstore.AtomVectorStore avs

        self.N += 1
        itr = vec.mydict.begin()
        end = vec.mydict.end()
        while(not itr.eq(end)):
            a = itr.first
            v = itr.second
            if has_a2avs(self.ii, a):
                avs = <atomvectorstore.AtomVectorStore> self.ii.ele(a)
            else:
                avs = atomvectorstore.AtomVectorStore()
                set_a2avs(self.ii, a, <void*> avs)

            avs.add(vec)
            itr.advance()
        return self

    # perhaps this should be made efficient
    cpdef int getDF(self, int a):
        cdef atomvectorstore.AtomVectorStore avs
        avs = self.getii(a)
        if avs is None:
            return 0
        else:
            return avs.N

    cpdef int getN(self):
        return self.N

    def __len__(self):
        self.ii.size()

    cpdef atomvectorstore.AtomVectorStore getii(self, int a):
        if has_a2avs(self.ii, a):
            return <atomvectorstore.AtomVectorStore> self.ii.ele(a)
        else:
            return None

    def clear(self, a):
        if has_a2avs(self.ii, a):
            avs = atomvectorstore.AtomVectorStore()
            set_a2avs(self.ii, a, <void*> avs)

        return self

    def atoms(self):
        cdef a2avsitr it = self.ii.begin()
        cdef a2avsitr end = self.ii.end()
        ret = []
        while (it.neq(end)):
            ret.append(<int> it.first)
            it.advance()
        return ret

    def __repr__(self):
        return "<InvertedIndex  N=%d  Terms=%d>" % (self.N, self.ii.size())
