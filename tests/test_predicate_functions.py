import unittest

from dedupe import predicate_functions as fn
from dedupe.cpredicates import ngrams


class TestPredicateFunctions(unittest.TestCase):
    def test_whole_field_predicate(self):
        assert fn.wholeFieldPredicate("donald") == {"donald"}
        assert fn.wholeFieldPredicate("go-of,y  ") == {"go-of,y  "}
        assert fn.wholeFieldPredicate(" cip ciop ") == {" cip ciop "}

    def test_token_field_predicate(self):
        assert fn.tokenFieldPredicate("donald") == {"donald"}
        assert fn.tokenFieldPredicate("do\nal d") == {"do", "al", "d"}
        assert fn.tokenFieldPredicate("go-of y  ") == {"go", "of", "y"}
        assert fn.tokenFieldPredicate(" cip   ciop ") == {"cip", "ciop"}

    def test_first_token_predicate(self):
        assert fn.firstTokenPredicate("donald") == {"donald"}
        assert fn.firstTokenPredicate("don ald") == {"don"}
        assert fn.firstTokenPredicate("do\nal d") == {"do"}
        assert fn.firstTokenPredicate("go-of y  ") == {"go"}
        assert fn.firstTokenPredicate(" cip   ciop ") == frozenset()

    def test_two_tokens_predicate(self):
        assert fn.firstTwoTokensPredicate("donald") == frozenset()
        assert fn.firstTwoTokensPredicate("don ald") == {"don ald"}
        assert fn.firstTwoTokensPredicate("do\nal d") == {"do\nal"}
        assert fn.firstTwoTokensPredicate("go-of y  ") == {"go-of"}
        assert fn.firstTwoTokensPredicate(" cip   ciop ") == frozenset()

    def test_common_integer_predicate(self):
        assert fn.commonIntegerPredicate("don4ld") == {"4"}
        assert fn.commonIntegerPredicate("donald 1992") == {"1992"}
        assert fn.commonIntegerPredicate("g00fy  ") == {"0"}
        assert fn.commonIntegerPredicate(" c1p   c10p ") == {"1", "10"}

    def test_alpha_numeric_predicate(self):
        assert fn.alphaNumericPredicate("don4ld") == {"don4ld"}
        assert fn.alphaNumericPredicate("donald 1992") == {"1992"}
        assert fn.alphaNumericPredicate("g00fy  ") == {"g00fy"}
        assert fn.alphaNumericPredicate(" c1p   c10p ") == {"c1p", "c10p"}

    def test_near_integers_predicate(self):
        assert fn.nearIntegersPredicate("don4ld") == {"3", "4", "5"}
        assert fn.nearIntegersPredicate("donald 1992") == {"1991", "1992", "1993"}
        assert fn.nearIntegersPredicate("g00fy  ") == {"-1", "0", "1"}
        assert fn.nearIntegersPredicate(" c1p   c10p ") == {
            "0",
            "1",
            "2",
            "9",
            "10",
            "11",
        }

    def test_hundred_integers_predicate(self):
        assert fn.hundredIntegerPredicate("don456ld") == {"400"}
        assert fn.hundredIntegerPredicate("donald 1992") == {"1900"}
        assert fn.hundredIntegerPredicate("g00fy  ") == {"00"}
        assert fn.hundredIntegerPredicate(" c111p   c1230p ") == {"100", "1200"}

    def test_hundred_integers_odd_predicate(self):
        assert fn.hundredIntegersOddPredicate("don456ld") == {"400"}
        assert fn.hundredIntegersOddPredicate("donald 1991") == {"1901"}
        assert fn.hundredIntegersOddPredicate("g00fy  ") == {"00"}
        assert fn.hundredIntegersOddPredicate(" c111p   c1230p ") == {"101", "1200"}

    def test_first_integer_predicate(self):
        assert fn.firstIntegerPredicate("donald 456") == frozenset()
        assert fn.firstIntegerPredicate("1992 donald") == {"1992"}
        assert fn.firstIntegerPredicate("00fy  ") == {"00"}  # !!!
        assert fn.firstIntegerPredicate("111 p   c1230p ") == {"111"}

    def test_common_two_tokens(self):
        assert fn.commonTwoTokens("d on 456 ld") == {"d on", "on 456", "456 ld"}
        assert fn.commonTwoTokens("donald 1992") == {"donald 1992"}
        assert fn.commonTwoTokens("g00fy  ") == frozenset()
        assert fn.commonTwoTokens(" c1p   c10p ") == {"c1p c10p"}

    def test_common_three_tokens(self):
        assert fn.commonThreeTokens("d on 456 ld") == {"d on 456", "on 456 ld"}
        assert fn.commonThreeTokens("donald 1992") == frozenset()
        assert fn.commonThreeTokens("g00fy  ") == frozenset()
        assert fn.commonThreeTokens(" c1p   c10p  c100p") == {"c1p c10p c100p"}

    def test_fingerprint(self):
        assert fn.fingerprint("don 456 ld ") == {"456donld"}
        assert fn.fingerprint("donald 1991") == {"1991donald"}
        assert fn.fingerprint(" g00fy  ") == {"g00fy"}
        assert fn.fingerprint(" c11p   c10p ") == {"c10pc11p"}

    def test_one_gram_fingerprint(self):
        assert fn.oneGramFingerprint("don 456 ld") == {"456dlno"}
        assert fn.oneGramFingerprint("donald 1992") == {"129adlno"}
        assert fn.oneGramFingerprint(" g00fy  ") == {"0fgy"}
        assert fn.oneGramFingerprint(" c1p   c10p ") == {"01cp"}

        def prevImpl(field: str):
            return {"".join(sorted(set(ngrams(field.replace(" ", ""), 1)))).strip()}

        assert fn.oneGramFingerprint("don 456 ld"), prevImpl("don 456 ld")
        assert fn.oneGramFingerprint("donald 1992"), prevImpl("donald 1992")
        assert fn.oneGramFingerprint(" g00fy  "), prevImpl(" g00fy  ")
        assert fn.oneGramFingerprint(" c1p   c10p "), prevImpl(" c1p   c10p ")

    def test_two_gram_fingerprint(self):
        assert fn.twoGramFingerprint("don4ld") == {"4ldoldn4on"}
        assert fn.twoGramFingerprint("donald 1992") == {"199299ald1doldnaon"}
        assert fn.twoGramFingerprint("g00fy  ") == {"000ffyg0"}
        assert fn.twoGramFingerprint(" c1p   c10p ") == {"0p101pc1pc"}
        assert fn.twoGramFingerprint("7") == frozenset()

        def prevImpl(field: str):
            if len(field) > 1:
                return frozenset(
                    (
                        "".join(
                            sorted(
                                gram.strip()
                                for gram in set(ngrams(field.replace(" ", ""), 2))
                            )
                        ),
                    )
                )
            else:
                return frozenset()

        assert fn.twoGramFingerprint("don4ld") == prevImpl("don4ld")
        assert fn.twoGramFingerprint("donald 1992") == prevImpl("donald 1992")
        assert fn.twoGramFingerprint("g00fy") == prevImpl("g00fy")
        assert fn.twoGramFingerprint(" c1p   c10p "), prevImpl(" c1p   c10p ")
        assert fn.twoGramFingerprint("a") == prevImpl("a")

    def test_common_four_gram(self):
        assert fn.commonFourGram("don4ld") == {"don4", "on4l", "n4ld"}
        assert fn.commonFourGram("donald 1992") == {
            "dona",
            "onal",
            "nald",
            "ald1",
            "ld19",
            "d199",
            "1992",
        }
        assert fn.commonFourGram("g00fy  ") == {"g00f", "00fy"}
        assert fn.commonFourGram(" c1p   c10p ") == {"c1pc", "1pc1", "pc10", "c10p"}

    def test_common_six_gram(self):
        assert fn.commonSixGram("don4ld") == {"don4ld"}
        assert fn.commonSixGram("donald 1992") == {
            "donald",
            "onald1",
            "nald19",
            "ald199",
            "ld1992",
        }
        assert fn.commonSixGram("g00fy  ") == frozenset()
        assert fn.commonSixGram(" c1p   c10p ") == {"c1pc10", "1pc10p"}

    def test_same_three_char_start_predicate(self):
        assert fn.sameThreeCharStartPredicate("don4ld") == {"don"}
        assert fn.sameThreeCharStartPredicate("donald 1992") == {"don"}
        assert fn.sameThreeCharStartPredicate("g00fy  ") == {"g00"}
        assert fn.sameThreeCharStartPredicate(" c1p   c10p ") == {"c1p"}

    def test_same_five_char_start_predicate(self):
        assert fn.sameFiveCharStartPredicate("don4ld") == {"don4l"}
        assert fn.sameFiveCharStartPredicate("donald 1992") == {"donal"}
        assert fn.sameFiveCharStartPredicate("g00fy  ") == {"g00fy"}
        assert fn.sameFiveCharStartPredicate(" c1p   c10p ") == {"c1pc1"}

    def test_same_seven_char_start_predicate(self):
        assert fn.sameSevenCharStartPredicate("don4ld") == {"don4ld"}
        assert fn.sameSevenCharStartPredicate("donald 1992") == {"donald1"}
        assert fn.sameSevenCharStartPredicate("g00fy  ") == {"g00fy"}
        assert fn.sameSevenCharStartPredicate(" c1p   c10p ") == {"c1pc10p"}

    def test_suffix_array(self):
        assert fn.suffixArray("don4ld") == {"don4ld", "on4ld"}
        assert fn.suffixArray("donald 1992") == {
            "donald 1992",
            "onald 1992",
            "nald 1992",
            "ald 1992",
            "ld 1992",
            "d 1992",
            " 1992",
        }
        assert fn.suffixArray("g00fy  ") == {"g00fy  ", "00fy  ", "0fy  "}
        assert fn.suffixArray(" c1p\nc10p ") == {
            " c1p\nc10p ",
            "c1p\nc10p ",
            "1p\nc10p ",
            "p\nc10p ",
            "\nc10p ",
            "c10p ",
        }

    def test_sorted_acronym(self):
        assert fn.sortedAcronym("don 4l d") == {"4dd"}
        assert fn.sortedAcronym("donald 19 92") == {"19d"}
        assert fn.sortedAcronym("g 00f y  ") == {"0gy"}
        assert fn.sortedAcronym(" c1p   c10p ") == {"cc"}

    def test_double_metaphone(self):
        assert fn.doubleMetaphone("i") == {"A"}
        assert fn.doubleMetaphone("donald") == {"TNLT"}
        assert fn.doubleMetaphone("goofy") == {"KF"}
        assert fn.doubleMetaphone("cipciop") == {"SPSP", "SPXP"}

    def test_metaphone_token(self):
        assert fn.metaphoneToken("i") == {"A"}
        assert fn.metaphoneToken("don ald") == {"TN", "ALT"}
        assert fn.metaphoneToken("goo fy") == {"K", "F"}
        assert fn.metaphoneToken("cip ciop") == {"SP", "XP"}

    def test_whole_set_predicate(self):
        assert fn.wholeSetPredicate({"i"}) == {r"{'i'}"}
        assert fn.wholeSetPredicate({"donald"}) == {r"{'donald'}"}
        assert fn.wholeSetPredicate({"goofy"}) == {r"{'goofy'}"}
        assert fn.wholeSetPredicate({"cipciop"}) == {r"{'cipciop'}"}

    # TODO: test commonSetElementPredicate
    # TODO: test commonTwoElementsPredicate
    # TODO: test commonThreeElementsPredicate
    # TODO: test lastSetElementPredicate
    # TODO: test firstSetElementPredicate

    def test_magnitude_of_cardinality(self):
        assert fn.magnitudeOfCardinality(range(0)) == frozenset()
        assert fn.magnitudeOfCardinality(range(98)) == {"2"}
        assert fn.magnitudeOfCardinality(range(100)) == {"2"}
        assert fn.magnitudeOfCardinality(range(10**6)) == {"6"}

    def test_lat_long_grid_predicate(self):
        assert fn.latLongGridPredicate((1.11, 2.22)) == {"(1.1, 2.2)"}
        assert fn.latLongGridPredicate((1.11, 2.27)) == {"(1.1, 2.3)"}
        assert fn.latLongGridPredicate((1.18, 2.22)) == {"(1.2, 2.2)"}
        assert fn.latLongGridPredicate((1.19, 2.29)) == {"(1.2, 2.3)"}

    def test_predicates_correctness(self):
        field = "123 16th st"
        assert fn.sortedAcronym(field) == {"11s"}
        assert fn.wholeFieldPredicate(field) == {"123 16th st"}
        assert fn.firstTokenPredicate(field) == {"123"}
        assert fn.firstTokenPredicate("") == frozenset()
        assert fn.firstTokenPredicate("123/") == {"123"}
        assert fn.firstTwoTokensPredicate(field) == {"123 16th"}
        assert fn.firstTwoTokensPredicate("oneword") == frozenset()
        assert fn.firstTwoTokensPredicate("") == frozenset()
        assert fn.firstTwoTokensPredicate("123 456/") == {"123 456"}
        assert fn.tokenFieldPredicate(" ") == frozenset()
        assert fn.tokenFieldPredicate(field) == {"123", "16th", "st"}
        assert fn.commonIntegerPredicate(field) == {"123", "16"}
        assert fn.commonIntegerPredicate("foo") == frozenset()
        assert fn.firstIntegerPredicate("foo") == frozenset()
        assert fn.firstIntegerPredicate("1foo") == {"1"}
        assert fn.firstIntegerPredicate("f1oo") == frozenset()
        assert fn.sameThreeCharStartPredicate(field) == {"123"}
        assert fn.sameThreeCharStartPredicate("12") == {"12"}
        assert fn.commonFourGram("12") == frozenset()
        assert fn.sameFiveCharStartPredicate(field) == {"12316"}
        assert fn.sameSevenCharStartPredicate(field) == {"12316th"}
        assert fn.nearIntegersPredicate(field) == {
            "15",
            "17",
            "16",
            "122",
            "123",
            "124",
        }
        assert fn.commonFourGram(field) == {
            "1231",
            "2316",
            "316t",
            "16th",
            "6ths",
            "thst",
        }
        assert fn.commonSixGram(field) == {"12316t", "2316th", "316ths", "16thst"}
        assert fn.initials(field, 12) == {"123 16th st"}
        assert fn.initials(field, 7) == {"123 16t"}
        assert fn.ngrams(field, 3) == [
            "123",
            "23 ",
            "3 1",
            " 16",
            "16t",
            "6th",
            "th ",
            "h s",
            " st",
        ]
        assert fn.unique_ngrams(field, 3) == {
            "123",
            "23 ",
            "3 1",
            " 16",
            "16t",
            "6th",
            "th ",
            "h s",
            " st",
        }
        assert fn.commonTwoElementsPredicate((1, 2, 3)) == {"1 2", "2 3"}
        assert fn.commonTwoElementsPredicate((1,)) == frozenset()
        assert fn.commonThreeElementsPredicate((1, 2, 3)) == {"1 2 3"}
        assert fn.commonThreeElementsPredicate((1,)) == frozenset()

        assert fn.fingerprint("time sandwich") == {"sandwichtime"}
        assert fn.oneGramFingerprint("sandwich time") == {"acdehimnstw"}
        assert fn.twoGramFingerprint("sandwich time") == {"anchdwhticimmendsatiwi"}
        assert fn.twoGramFingerprint("1") == frozenset()
        assert fn.commonTwoTokens("foo bar") == {"foo bar"}
        assert fn.commonTwoTokens("foo") == frozenset()
