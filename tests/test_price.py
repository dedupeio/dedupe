import unittest
import dedupe
from dedupe.fieldclasses import PriceType

class TestPrice(unittest.TestCase):
    def test_comparator(self) :
        assert PriceType.comparator(1, 10) == 1
        assert PriceType.comparator(10, 1) == 1


