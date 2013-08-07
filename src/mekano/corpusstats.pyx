cimport dedupe.mekano.atomvector as atomvector
import cPickle

cdef class CorpusStats:
    """
    Maintains DF and N for atoms.

    This class is meant for sequential processing of vectors.
    It does not check if the same vector has been passed twice.

    cs = CorpusStats()
    cs.add(atomvector)
    
    cs.getN()
    cs.getDF(atoms)
    
    Supports iteritems()
    """

    def __init__(self):
        self.df = atomvector.AtomVector()
        self.N = 0
    
    def __reduce__(self):
        return CorpusStats, (), self.__getstate__(), None, None
    
    def __getstate__(self):
        return [self.df, self.N]
    
    def __setstate__(self, s):
        self.df, self.N = s

    cpdef add(self, atomvector.AtomVector vec):
        """
        add(av)

        Processes the atoms in av and updates DF and N
        av is assumed to not have duplicate atoms.
        """
        cdef int a
        cdef atomvector.dictitr itr, end

        self.N += 1
        itr = vec.mydict.begin()
        end = vec.mydict.end()
        while(itr.neq(end)):
            a = itr.first
            #self.df[a] = self.df[a] + 1
            self.df.set(a, self.df.get(a)+1)
            itr.advance()

    def __repr__(self):
        return "<CorpusStats   %d atoms, %d docs>" % (len(self.df), self.N)

    cpdef int getDF(self, int atom):
        return <int> self.df.get(atom)

    cpdef int getN(self):
        return self.N
    
    def iteritems(self):
        return self.df.iteritems()

    def save(self, filename):
        with open(filename, "w") as fout:
            cPickle.dump(self, fout, -1)

    @classmethod
    def fromfile(cls, filename):
        with open(filename, "r") as fin:
            a = cPickle.load(fin)
        return a
    
