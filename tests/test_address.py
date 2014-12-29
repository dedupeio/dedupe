import unittest
import dedupe
from dedupe.variables.address import USAddressType
import numpy

class TestPrice(unittest.TestCase):
    def test_comparator(self) :
        us = USAddressType({'field' : 'foo'})
        numpy.testing.assert_almost_equal(us.comparator('123 E Main St', 
                                                        '124 W Main St'),
            numpy.array([1, 1, 1, 0, 0, 2.16666675,
                         5.5, 0.5, 0.5, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 1, 1, 1, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0]))

        numpy.testing.assert_almost_equal(us.comparator('po box 31', 
                                                        '124 W Main St'),
            numpy.array([1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0]))
                                
        numpy.testing.assert_almost_equal(us.comparator('po box 31', 
                                                        'po box 41'),
            numpy.array([1, 1, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0.5, 3, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 1, 0,
                         0, 0, 0, 0, 0]))

        numpy.testing.assert_almost_equal(us.comparator('69th and main st', 
                                                        '70th and main st'),
            numpy.array([1, 1, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 3, 0, 0,
                         0.5, 0.5, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 1, 1]))

        numpy.testing.assert_equal(us.comparator('69th and main st', 
                                                 '70th and main st'),
                                   us.comparator('main st and 69th',
                                                 '70th and main st'))


        



def prettyPrint(us, comparison) :
    for e in zip(us.higher_vars, comparison) :
        print "%s:\t %s" % e

