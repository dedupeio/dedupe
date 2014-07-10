import unittest
import dedupe
import dedupe.centroid
import numpy
import random
import warnings


class CanonicalizationTest(unittest.TestCase) :

	def setUp(self) :
		self.data_d = { 1 : {"name": "mary crane", "address": "123 main st", "zip":"12345"}, 
					 	2 : {"name": "mary crane east", "address": "123 main street", "zip":""}, 
						3 : {"name": "mary crane west", "address": "123 man st", "zip":""}}
		deduper = dedupe.Dedupe({'name': {'type': 'String'},
                      			 'address':   {'type': 'String'},
                      			 'zip': {'type': 'String'}})
		self.data_model = deduper.data_model


	def test_get_centroid(self) :
		from dedupe.distance.affinegap import normalizedAffineGapDistance as comparator
		attributeList = ['mary crane center', 'mary crane center north', 'mary crane league - mary crane - west', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league - mary crane - east', 'mary crane family and day care center', 'mary crane west', 'mary crane center east', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league', 'mary crane', 'mary crane east 0-3', 'mary crane north', 'mary crane north 0-3', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east']
		centroid = dedupe.centroid.getCentroid (attributeList, comparator)
		assert centroid == 'mary crane'

	def test_get_canonical_rep(self) :
		rep = dedupe.centroid.getCanonicalRep((1,2,3), self.data_d, self.data_model)
		assert rep == {'name': 'mary crane', 'address': '123 main street', 'zip':"12345"}

		rep = dedupe.centroid.getCanonicalRep((1,2), self.data_d, self.data_model)
		assert rep == {"name": "mary crane", "address": "123 main st", "zip":"12345"}

		rep = dedupe.centroid.getCanonicalRep((1,), self.data_d, self.data_model)
		assert rep == {"name": "mary crane", "address": "123 main st", "zip":"12345"}


if __name__ == "__main__":
	unittest.main()
