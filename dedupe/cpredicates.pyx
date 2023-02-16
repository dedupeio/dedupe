# cython: c_string_type=unicode, c_string_encoding=utf8, infertypes=True, language_level=3

cpdef list ngrams(basestring field, int n):
    """ngrams returns all contiguous sequences of n characters
    of a given field.

    :param field: the string to be sequenced
    :param n: the number of characters to be included in each gram

    usage:
    >>> from dedupe.dedupe.predicated import ngrams
    >>> ngrams("deduplicate", 3)
    ['ded', 'edu', 'dup', 'upl', 'pli', 'lic', 'ica', 'cat', 'ate']
    """
    cdef unicode ufield = _ustring(field)

    cdef int i
    cdef int n_char = len(ufield)
    cdef int n_grams = n_char - n + 1
    cdef list grams = [ufield[i:i+n] for i in range(n_grams)]
    return grams


cpdef frozenset unique_ngrams(basestring field, int n):
    """unique_ngrams returns all contiguous unique sequences of n characters
    of a given field.

    :param field: the string to be sequenced
    :param n: the number of characters to be included in each gram

    usage:
    >>> from dedupe.dedupe.predicated import unique_ngrams
    >>> unique_ngrams("mississippi", 2)
    {"mi", "is", "ss", "si", "ip", "pp", "pi"}
    """
    cdef unicode ufield = _ustring(field)

    cdef int i
    cdef int n_char = len(ufield)
    cdef int n_grams = n_char - n + 1
    cdef set grams = {ufield[i:i+n] for i in range(n_grams)}
    return frozenset(grams)


cpdef frozenset initials(basestring field, int n):
    """returns a tuple containing the first n chars of a field.
    The whole field is returned if n is greater than the field length.

    :param field: the string
    :type n: int

    usage:
    >>> initials("dedupe", 7)
    ('dedupe', )
    >>> initials("deduplication", 7)
    ('dedupli', )
    """
    cdef unicode ufield = _ustring(field)

    return frozenset((ufield[:n],))


cdef unicode _ustring(basestring s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    else : # safe because of basestring
        return <char *>s
