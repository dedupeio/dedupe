#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from libc.stdlib cimport malloc, free

def split_class(char *class_str, delim):
    out = set(class_str.split(delim))
    return out

cpdef float jaccard(sl, sr):
    set_union = sl.union(sr)
    set_intersect = sl.intersection(sr)
    if len(set_union) == 0:
        return 0.0
    return len(sl & sr) / float(len(set_union))

cpdef float compareJaccard(char *s1, char *s2, delim='**'):
    s1_class = split_class(s1, delim)
    s2_class = split_class(s2, delim)
    out = jaccard(s1_class, s2_class)
    return out
