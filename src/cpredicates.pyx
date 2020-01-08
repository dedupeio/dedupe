# cython: c_string_type=unicode, c_string_encoding=utf8, infertypes=True, language_level=3

cpdef list ngrams(basestring field, int n):
    """ngrams returns all unique, contiguous sequences of n characters
    of a given field.
        
    :param field: the string to be 
    :param n: the number of characters to be included in each gram
    
    usage:
    >>> from dedupe.dedupe.predicated import ngrams
    >>> ngrams("deduplicate", 3)
    ('ded', 'edu', 'dup', 'upl', 'pli', 'lic', 'ica', 'cat', 'ate')
    """
    cdef unicode ufield = _ustring(field)

    cdef list grams = []
    cdef int i, j
    cdef int n_char = len(ufield)
    for i in range(n_char):
        for j in range(i+n, min(n_char, i+n)+1):
            grams.append(ufield[i:j])
            
    return grams

cpdef tuple initials(basestring field, int n):
    """predicate which returns first a tuple containing
    the first n chars of a field if and only if the
    field contains at least n characters, or an empty
    tuple otherwise.
    
    :param field: the string 
    :type n: int, default None
    
    usage:
    >>> initials("dedupe", 7)
    ('dedupe', )
    >>> initials("deduplication", 7)
    ('dedupli', )
    """
    cdef unicode ufield = _ustring(field)

    return (ufield[:n], )



cdef unicode _ustring(basestring s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    else : # safe because of basestring
        return <char *>s
