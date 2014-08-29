import dedupe
import unittest

class ParsingTest(unittest.TestCase) :
    def setUp(self) :
        self.index = dedupe.tfidf.TfIdfIndex('foo')
        
    def test_keywords(self) :
        assert self.index._parseTerms(u'`AND OR NOT ( ) ATOM \\EOF, xx') ==\
            ['xxAND', 'xxOR', 'xxNOT', 'xxATOM', 'xxEOF', 'xx']
        
        

if __name__ == "__main__":
    unittest.main()
