import dedupe
import unittest

class ParsingTest(unittest.TestCase) :
    def setUp(self) :
        self.index = dedupe.tfidf.TfIdfIndex('foo')
        
    def test_keywords(self) :
        assert self.index._parseTerms(u'`AND OR NOT ( ) ATOM \\EOF') ==\
            ['\AND', '\OR', '\NOT', '\ATOM', '\EOF']
        
        self.index.index(1, 'AND OR EOF NOT')
        assert self.index.search('AND OR EOF NOT')[0] == 1

    def test_keywords_title(self) :
        assert self.index._parseTerms(u'`And Or Not ( ) Atom \\Eof') ==\
            ['\AND', '\OR', '\NOT', '\ATOM', '\EOF']
        
        self.index.index(1, 'And Or Eof Not')
        assert self.index.search('And Or Eof Not')[0] == 1


    def test_empty_search(self) :
        assert self.index.search('') == []

if __name__ == "__main__":
    unittest.main()
