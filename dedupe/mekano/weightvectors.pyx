cimport atomvector
cimport corpusstats

cdef extern from "math.h":
    double log(double x)
    
cdef class WeightVectors:
    """For creating LTC vectors.

        >>> wv = WeightVectors(cs, cache=False) 

    Creates a weight-vector cache linked to the given L{CorpusStats} object.

        >>> cs.add(unweighted_vector)
        >>> weighted_vector = wv[unweighted_vector]
        >>> assert weighted_vector.CosineLen() == 1.0

    """
    
    cdef double n
    cdef corpusstats.CorpusStats cs
    cdef object cache
    cdef int maintaincache
    cdef int n_access
    cdef int n_hits
    
    def __init__(self, corpusstats.CorpusStats cs, cache=False):
        """Create a WeightVector object.
        
        @param cs       : The L{CorpusStats} object that this should link to
        @param cache    : Whether to maintain a cache for fast lookup
        """
        self.cs = cs
        self.cache = {}
        self.n_access = 0
        self.n_hits = 0
        if cache:
            self.maintaincache = 1
        else:
            self.maintaincache = 0

    def __getitem__(self, atomvector.AtomVector vec):
        cdef double n
        cdef int a
        cdef double v
        cdef atomvector.dictitr itr, end
        cdef atomvector.AtomVector wav
        cdef int df

        self.n_access += 1
        if self.maintaincache == 1 and vec in self.cache:
            self.n_hits += 1
            return self.cache[vec]
        else:
            wav = atomvector.AtomVector()
            n = self.cs.getN()

            itr = vec.mydict.begin()
            end = vec.mydict.end()
            while(itr.neq(end)):
                a = itr.first
                v = itr.second
                df = self.cs.getDF(a)
                if df > 0:
                    wav.set(a, (1.0+log(v))*log((1.0+n)/df))
                itr.advance()
            wav.Normalize()
            if self.maintaincache == 1:
                self.cache[vec] = wav
            return wav

    def __repr__(self):
        return "<WeightVectors: #access:%d #hits:%d %s>" % (self.n_access, self.n_hits, repr(self.cs))
