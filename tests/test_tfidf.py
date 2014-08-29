import dedupe
import unittest

class ParsingTest(unittest.TestCase) :
    def setUp(self) :
        self.index = dedupe.tfidf.TfIdfIndex('foo')
        
    def test_keywords(self) :
        assert self.index._parseTerms(u'`AND OR NOT ( ) ATOM \\EOF, xx') ==\
            ['xxAND', 'xxOR', 'xxNOT', 'xxATOM', 'xxEOF', 'xx']
        
        self.index.index(1, 'AND OR EOF NOT')
        assert self.index.search('AND OR EOF NOT')[0] == 1

if __name__ == "__main__":
    unittest.main()
