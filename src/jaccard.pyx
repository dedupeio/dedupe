#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from libc.stdlib cimport malloc, free

cpdef split_class_string(char *setstr, char *delim):
    out = setstr.split(delim)
    return set(out)

cpdef float jaccard(sl, sr):
    set_union = sl.union(sr)
    set_intersect = sl.intersection(sr)
    if len(set_union) == 0:
        return 0.0
    return len(sl & sr) / float(len(set_union))

cpdef float compareJaccard(char *setstr1, char *setstr2, delim='**'):
    s1 = split_class_string(setstr1, delim)
    s2 = split_class_string(setstr2, delim)
    out = jaccard(s1, s2)
    return out
