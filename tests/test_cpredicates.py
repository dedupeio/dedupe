import unittest

from dedupe.cpredicates import initials, ngrams, unique_ngrams


class TestCPredicates(unittest.TestCase):
    def test_ngrams(self):
        assert ngrams("deduplicate", 1) == [
            "d",
            "e",
            "d",
            "u",
            "p",
            "l",
            "i",
            "c",
            "a",
            "t",
            "e",
        ]
        assert ngrams("deduplicate", 2) == [
            "de",
            "ed",
            "du",
            "up",
            "pl",
            "li",
            "ic",
            "ca",
            "at",
            "te",
        ]
        assert ngrams("deduplicate", 3) == [
            "ded",
            "edu",
            "dup",
            "upl",
            "pli",
            "lic",
            "ica",
            "cat",
            "ate",
        ]
        assert ngrams("deduplicate", 4) == [
            "dedu",
            "edup",
            "dupl",
            "upli",
            "plic",
            "lica",
            "icat",
            "cate",
        ]
        assert ngrams("deduplicate", 5) == [
            "dedup",
            "edupl",
            "dupli",
            "uplic",
            "plica",
            "licat",
            "icate",
        ]
        assert ngrams("deduplicate", 6) == [
            "dedupl",
            "edupli",
            "duplic",
            "uplica",
            "plicat",
            "licate",
        ]
        assert ngrams("deduplicate", 7) == [
            "dedupli",
            "eduplic",
            "duplica",
            "uplicat",
            "plicate",
        ]
        assert ngrams("deduplicate", 8) == [
            "deduplic",
            "eduplica",
            "duplicat",
            "uplicate",
        ]
        assert ngrams("deduplicate", 9) == ["deduplica", "eduplicat", "duplicate"]
        assert ngrams("deduplicate", 10) == ["deduplicat", "eduplicate"]
        assert ngrams("deduplicate", 11) == ["deduplicate"]
        assert ngrams("deduplicate", 12) == []
        assert ngrams("deduplicate", 100) == []

    def test_unique_ngrams(self):
        assert unique_ngrams("mississippi", 1) == {"m", "i", "s", "p"}
        assert unique_ngrams("mississippi", 2) == {
            "mi",
            "is",
            "ss",
            "si",
            "ip",
            "pp",
            "pi",
        }
        assert unique_ngrams("mississippi", 3) == {
            "mis",
            "iss",
            "ssi",
            "sis",
            "sip",
            "ipp",
            "ppi",
        }
        assert unique_ngrams("mississippi", 4) == {
            "miss",
            "issi",
            "ssis",
            "siss",
            "ssip",
            "sipp",
            "ippi",
        }
        assert unique_ngrams("mississippi", 5) == {
            "missi",
            "issis",
            "ssiss",
            "sissi",
            "issip",
            "ssipp",
            "sippi",
        }
        assert unique_ngrams("mississippi", 6) == {
            "missis",
            "ississ",
            "ssissi",
            "sissip",
            "issipp",
            "ssippi",
        }
        assert unique_ngrams("mississippi", 7) == {
            "mississ",
            "ississi",
            "ssissip",
            "sissipp",
            "issippi",
        }
        assert unique_ngrams("mississippi", 8) == {
            "mississi",
            "ississip",
            "ssissipp",
            "sissippi",
        }
        assert unique_ngrams("mississippi", 9) == {
            "mississip",
            "ississipp",
            "ssissippi",
        }
        assert unique_ngrams("mississippi", 10) == {"mississipp", "ississippi"}
        assert unique_ngrams("mississippi", 11) == {"mississippi"}
        assert unique_ngrams("mississippi", 12) == set()
        assert unique_ngrams("mississippi", 100) == set()

    def test_initials(self):
        assert initials("deduplicate", 1) == {"d"}
        assert initials("deduplicate", 2) == {"de"}
        assert initials("deduplicate", 3) == {"ded"}
        assert initials("deduplicate", 4) == {"dedu"}
        assert initials("deduplicate", 5) == {"dedup"}
        assert initials("deduplicate", 6) == {"dedupl"}
        assert initials("deduplicate", 7) == {"dedupli"}
        assert initials("deduplicate", 8) == {"deduplic"}
        assert initials("deduplicate", 9) == {"deduplica"}
        assert initials("deduplicate", 10) == {"deduplicat"}
        assert initials("deduplicate", 11) == {"deduplicate"}
        assert initials("deduplicate", 12) == {"deduplicate"}
        assert initials("deduplicate", 100) == {"deduplicate"}


if __name__ == "__main__":
    unittest.main()
