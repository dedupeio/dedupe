import unittest
import dedupe
import dedupe.centroid
import numpy
import random
import warnings

from dedupe.distance.affinegap import normalizedAffineGapDistance as comparator

class CanonicalizationTest(unittest.TestCase) :
	def test_get_centroid(self) :
		attributeList = ['mary crane center', 'mary crane center north', 'mary crane league - mary crane - west', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league - mary crane - east', 'mary crane family and day care center', 'mary crane west', 'mary crane center east', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league', 'mary crane', 'mary crane east 0-3', 'mary crane north', 'mary crane north 0-3', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east']
		centroid = dedupe.centroid.getCentroid (attributeList, comparator)
		assert centroid == 'mary crane'

if __name__ == "__main__":
	unittest.main()
