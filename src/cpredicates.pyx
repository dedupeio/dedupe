cpdef ngrams(char *field, int n):
    """ngrams returns all unique, contiguous sequences of n characters
    of a given field.
        
    :param field: the string to be 
    :param n: the number of characters to be included in each gram
    
    usage:
    >>> from dedupe.dedupe.predicated import ngrams
    >>> ngrams("deduplicate", 3)
    ('ded', 'edu', 'dup', 'upl', 'pli', 'lic', 'ica', 'cat', 'ate')
    """
    cdef set grams = set([])
    cdef int i, j
    cdef int n_char = len(field)
    for i in xrange(n_char):
        for j in xrange(i+n, min(n_char, i+n)+1):
            grams.add(field[i:j])
            
    return grams

cpdef initials(char *field, int n):
    """predicate which returns first a tuple containing
    the first n chars of a field if and only if the
    field contains at least n characters, or an empty
    tuple otherwise.
    
    :param field: the string 
    :type n: int, default None
    
    usage:
    >>> initials("dedupe", 7)
    ()
    >>> initials("deduplication", 7)
    ('dedupli', )
    """
    return (field[:n], ) if len(field) > n-1 else () 



