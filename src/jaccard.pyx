#!python
#cython: boundscheck=False, wraparound=False


cpdef float jaccard(sl, sr):
    set_union = sl.union(sr)
    set_intersect = sl.intersection(sr)
    if len(set_union) == 0:
        return 0.0
    return len(sl & sr) / float(len(set_union))

cpdef float compareJaccard(s1, s2):
    out = jaccard(s1, s2)
    return out
