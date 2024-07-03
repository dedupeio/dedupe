import unittest

import numpy

from dedupe.variables.exists import ExistsType


class TestExists(unittest.TestCase):
    def test_comparator(self):
        var = ExistsType("foo")
        assert numpy.array_equal(var.comparator(None, None), [0, 0])
        assert numpy.array_equal(var.comparator(1, 1), [1, 0])
        assert numpy.array_equal(var.comparator(1, 0), [0, 1])

    def test_len_higher_vars(self):
        # The len > 1 is neccessary for the correct processing in datamodel.py
        var = ExistsType("foo")
        assert len(var) > 1
        assert len(var.higher_vars) > 1
        assert len(var) == len(var.higher_vars)
