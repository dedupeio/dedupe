"""atomvector module

The main functionality is provided by the L{AtomVector} class.

"""

cdef class AtomVector:
    """A dictionary-like object for representing sparse vectors.

    Initialization:
        >>> a = AtomVector(vec = None, name="")     # vec should support iteritems() returning int:float pairs.
        >>> a = AtomVector.fromstring("1:10.0 2:25.0 7:13.0")
        >>> a = b.copy()                            # another AtomVector
        >>> a.clear()

    Iterating (behaves like a C{defaultdict(float)}):
        >>> a.iterkeys()
        >>> a.iteritems()
        >>> for atom in a:
        ...     print atom

    Printing (using an L{AtomFactory}):
        >>> a.tostring(af)
    
    Operators:
        >>> a + b               # Merges two atomvectors
        >>> a * b               # Dot product
        >>> a / c               # Element-wise divide (c is a float)
    
    Length-related:
        >>> a.Normalize()       # in-place
        >>> a.Normalized()      # returns a copy.
        >>> a.CosineLeng()
    
    """

    def __cinit__(AtomVector self, vec = None, name = ""):
        self.name = name
        self.mydict = new_dict()

    def __init__(AtomVector self, vec = None, name = ""):
        if vec:
            for k, v in vec.iteritems():
                set_dict(self.mydict, k, v)

    def __dealloc__(AtomVector self):
        del_dict(self.mydict)


    def __reduce__(self):
        return AtomVector, (), self.__getstate__(), None, None

    def __getstate__(self):
        return (self.name, dict(self.iteritems()))

    def __setstate__(self, s):
        self.name = s[0]
        for k,v in s[1].iteritems():
            self.set(k,v)

    def __repr__(self):
        return "(" + self.name + "[" + ",".join(["%s:%5.3f" % (a,v) for a,v in self.iteritems()]) + "])"
        
    def tostring(self, af):
        """Return a pretty-formatted string.
        
        @param af       : An L{AtomFactory}
        @return         : A nicely formatted string, with "..." in case it's too long.
        """
        if len(self) < 50:
            return "(" + self.name + "[" + ",".join(["%s:%5.3f" % (af.get_object(a),v) for a,v in self.iteritems()]) + "])"
        else:
            s = sorted([(v,k) for k,v in self.iteritems()], reverse=True)
            largest = ",".join(["%s:%5.3f" % (af(a),v) for v,a in s[:5]])
            smallest = ",".join(["%s:%5.3f" % (af(a),v) for v,a in s[-5:]])
            return "(" + self.name + "[" + largest + "..." + smallest + "])"
            

    # >>>>>> Add
    def __iadd__(AtomVector self, AtomVector other):
        self.addvector(other)
        return self

    def __add__(AtomVector self, AtomVector other):
        cdef AtomVector ret = self.copy()
        merge_vectors(ret.mydict, other.mydict)
        return ret

    cpdef addvector(AtomVector self, AtomVector other):
        merge_vectors(self.mydict, other.mydict)

    def __mul__(AtomVector self, AtomVector other):
        return self.dot(other)
    
    cdef double dot(AtomVector self, AtomVector other):
        return get_dot(self.mydict, other.mydict)

    def __div__(AtomVector self, double d):
        cdef AtomVector ret = self.copy()
        ret._idiv(d)
        return ret

    def __idiv__(self, double d):
        self._idiv(d)
        return self

    cdef _idiv(self, double d):
        cdef double e
        cdef dictitr it = self.mydict.begin()
        cdef dictitr end = self.mydict.end()
        while (it.neq(end)):
            it.second /= d
            it.advance()

    cpdef CosineLen(AtomVector self):
        cdef double ret, e
        cdef dictitr it = self.mydict.begin()
        cdef dictitr end = self.mydict.end()
        ret = 0.0
        while (it.neq(end)):
            e = it.second
            ret += e*e
            it.advance()
        return sqrt(ret)

    cpdef Normalized(self):
        cdef AtomVector ret = self.copy()
        ret.Normalize()
        return ret

    cpdef Normalize(AtomVector self):
        cdef double denom = self.CosineLen()
        self._idiv(denom)

    # >>>>>> Get/Set
    def __getitem__(self, a):
        return self.get(a)

    def __setitem__(self, a, v):
        set_dict(self.mydict, a, v)

    cpdef set(AtomVector self, int a, double v):
        set_dict(self.mydict, a, v)

    cdef double get(AtomVector self, int a):
        # simply doing a .ele(a) will add a non-existant key
        cdef dictitr it = self.mydict.find(a)
        if it.eq(self.mydict.end()):
            return 0.0
        else:
            return it.second

    # >>>>>> Container behavior
    def __iter__(AtomVector self):
        return AtomVectorKeysIterator(self)

    def iterkeys(AtomVector self):
        return AtomVectorKeysIterator(self)

    def iteritems(AtomVector self):
        return AtomVectorItemsIterator(self)

    def __contains__(self, a):
        return self.contains(a)

    cpdef int contains(self, int a):
        return self.mydict.find(a).neq(self.mydict.end())

    def __len__(self):
        return self.mydict.size()

    def __delitem__(self, a):
        cdef dictitr it = self.mydict.find(a)
        if it.neq(self.mydict.end()):
            self.mydict.erase(it)

    # Creation/Copying
    cpdef copy(AtomVector self):
        cdef AtomVector ret = AtomVector()
        cdef dictitr it = self.mydict.begin()
        cdef dictitr end = self.mydict.end()
        while (it.neq(end)):
            ret.set(it.first, it.second)
            it.advance()
        return ret

    cpdef clear(AtomVector self):
        self.mydict.clear()

    @classmethod
    def fromstring(cls, str):
        cdef AtomVector av = AtomVector()
        cdef dicttype* d = av_fromstring(str)
        del_dict(av.mydict)
        av.mydict = d
        return av



cdef class AtomVectorKeysIterator:
    cdef dictitr itr
    cdef dictitr end

    def __init__(self, AtomVector av):
        self.itr = av.mydict.begin()
        self.end = av.mydict.end()

    def __iter__(self):
        return self

    def __next__(self):
        if self.itr.eq(self.end):
            raise StopIteration

        v = self.itr.first
        self.itr.advance()
        return v

cdef class AtomVectorItemsIterator:
    cdef dictitr itr
    cdef dictitr end

    def __init__(self, AtomVector av):
        self.itr = av.mydict.begin()
        self.end = av.mydict.end()

    def __iter__(self):
        return self

    def __next__(self):
        if self.itr.eq(self.end):
            raise StopIteration

        v = (self.itr.first, self.itr.second)
        self.itr.advance()
        return v
