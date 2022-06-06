import unittest

import dedupe.canonical


class CanonicalizationTest(unittest.TestCase):
    def test_get_centroid(self):
        from affinegap import normalizedAffineGapDistance as comparator

        attributeList = [
            "mary crane center",
            "mary crane center north",
            "mary crane league - mary crane - west",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league - mary crane - east",
            "mary crane family and day care center",
            "mary crane west",
            "mary crane center east",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league",
            "mary crane",
            "mary crane east 0-3",
            "mary crane north",
            "mary crane north 0-3",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
        ]

        centroid = dedupe.canonical.getCentroid(attributeList, comparator)
        assert centroid == "mary crane"

    def test_get_canonical_rep(self):
        record_list = [
            {"name": "mary crane", "address": "123 main st", "zip": "12345"},
            {"name": "mary crane east", "address": "123 main street", "zip": ""},
            {"name": "mary crane west", "address": "123 man st", "zip": ""},
        ]

        rep = dedupe.canonical.getCanonicalRep(record_list)
        assert rep == {
            "name": "mary crane",
            "address": "123 main street",
            "zip": "12345",
        }

        rep = dedupe.canonical.getCanonicalRep(record_list[0:2])
        assert rep == {"name": "mary crane", "address": "123 main st", "zip": "12345"}

        rep = dedupe.canonical.getCanonicalRep(record_list[0:1])
        assert rep == {"name": "mary crane", "address": "123 main st", "zip": "12345"}
