import unittest
import dedupe

class CanonicalizationTest(unittest.TestCase) :

	def test_get_centroid(self) :
		from dedupe.distance.affinegap import normalizedAffineGapDistance as comparator
		attributeList = ['mary crane center', 'mary crane center north', 'mary crane league - mary crane - west', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league - mary crane - east', 'mary crane family and day care center', 'mary crane west', 'mary crane center east', 'mary crane league mary crane center (east)', 'mary crane league mary crane center (north)', 'mary crane league mary crane center (west)', 'mary crane league', 'mary crane', 'mary crane east 0-3', 'mary crane north', 'mary crane north 0-3', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east', 'mary crane league - mary crane - west', 'mary crane league - mary crane - north', 'mary crane league - mary crane - east']
		centroid = dedupe.centroid.getCentroid (attributeList, comparator)
		assert centroid == 'mary crane'

	def test_get_canonical_rep(self) :
		record_list = [ {"name": "mary crane", "address": "123 main st", "zip":"12345"}, 
					 		 {"name": "mary crane east", "address": "123 main street", "zip":""}, 
							 {"name": "mary crane west", "address": "123 man st", "zip":""} ]
		rep = dedupe.centroid.getCanonicalRep((0,1,2), record_list)
		assert rep == {'name': 'mary crane', 'address': '123 main street', 'zip':"12345"}

		rep = dedupe.centroid.getCanonicalRep((0,1), record_list)
		assert rep == {"name": "mary crane", "address": "123 main st", "zip":"12345"}

		rep = dedupe.centroid.getCanonicalRep((0,), record_list)
		assert rep == {"name": "mary crane", "address": "123 main st", "zip":"12345"}

if __name__ == "__main__":
	unittest.main()
