import unittest

import numpy

from dedupe.variables.exists import ExistsType


class TestExists(unittest.TestCase):
    def test_comparator(self):
        var = ExistsType("foo")
        assert numpy.array_equal(var.comparator(None, None), [0, 0])
        assert numpy.array_equal(var.comparator(1, 1), [1, 0])
        assert numpy.array_equal(var.comparator(1, 0), [0, 1])
