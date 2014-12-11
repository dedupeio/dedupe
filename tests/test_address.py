import unittest
import dedupe
from dedupe.variables.address import USAddressType

class TestPrice(unittest.TestCase):
    def test_comparator(self) :
        us = USAddressType({'field' : 'foo'})
        prettyPrint(us, us.comparator('123 E Main St', '124 W Main St'))
        print
        prettyPrint(us, us.comparator('po box 31', '124 W Main St'))
        assert 1 == 0


def prettyPrint(us, comparison) :
    for e in zip(us.higher_vars, comparison) :
        print "%s:\t %s" % e
