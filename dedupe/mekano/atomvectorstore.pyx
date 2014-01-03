cimport atomvector
import atomvector

cdef class AtomVectorStore:
    """
    s = AtomVectorStore()

    """

    def __cinit__(AtomVectorStore self):
        self.mystore = new_store()
        self.N = 0

    def __dealloc__(AtomVectorStore self):
        del_store(self.mystore)
    
    
    def __reduce__(self):
        return AtomVectorStore, (), self.__getstate__(), None, None
    
    def __getstate__(self):
        return [av for av in self]
    
    def __setstate__(self, s):
        self.N = 0
        for av in s:
            self.add(av)

    cpdef add(AtomVectorStore self, atomvector.AtomVector av):
        add_store(self.mystore, <void*> av)
        self.N += 1

    def __repr__(self):
        return "<AtomVectorStore  %d documents>" % self.N

    def __iter__(AtomVectorStore self):
        self.itr =  0
        return self

    # extension types have __next__ instead of next()
    def __next__(AtomVectorStore self):
        cdef atomvector.AtomVector av

        if self.itr >= self.N:
            raise StopIteration

        av = <atomvector.AtomVector> self.mystore.ele(self.itr)
        self.itr += 1
        return av

    def __getitem__(self, int i):
        return self.getAt(i)

    cpdef atomvector.AtomVector getAt(self, int i):
        if i >= self.N:
            raise IndexError

        return <atomvector.AtomVector> self.mystore.ele(i)

    def __len__(self):
        return self.N
