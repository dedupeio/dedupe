cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "CUtils.h":

    # the const iterator
    ctypedef struct dictitr "IntDoubleMap::iterator":
        int first "operator->()->first"
        double second "operator->()->second"
        void advance "operator++" ()
        int eq "operator==" (dictitr o)
        int neq "operator!=" (dictitr o)

    # the hash map that internally stores the AtomVector
    ctypedef struct dicttype "IntDoubleMap":
        int size()
        void erase(dictitr i)
        dictitr find(int a)
        dictitr begin()
        dictitr end()
        int ele "operator[]" (int a)
        void clear()

    dicttype* new_dict "new IntDoubleMap" ()
    void del_dict "delete" (dicttype* h)
    void set_dict(dicttype* dict, int a, double v)

    char* make_str(dicttype* d)

    double get_dot(void* d1, void* d2)
    double cosine_dict(dicttype *d)
    void normalize_dict(dicttype *d)
    void merge_vectors(dicttype *v1, dicttype *v2)
    dicttype* av_fromstring(char *str)


cdef class AtomVector:

    cdef public object name

    cdef dicttype *mydict
    cdef dictitr itr

    cdef double get(AtomVector self, int a)
    cpdef set(AtomVector self, int a, double v)
    cpdef copy(AtomVector self)
    cpdef addvector(AtomVector self, AtomVector other)

    cpdef CosineLen(AtomVector self)
    cpdef Normalize(AtomVector self)
    cpdef Normalized(AtomVector self)

    cdef _idiv(AtomVector self, double d)

    cpdef clear(AtomVector self)

    cpdef int contains(AtomVector self, int a)
    
    cdef double dot (AtomVector self, AtomVector other)
