import dedupe
import unittest

class ParsingTest(unittest.TestCase) :
    def setUp(self) :
        self.index = dedupe.tfidf.TfIdfIndex('foo')
        
    def test_keywords(self) :
        assert self.index._parseTerms(u'`AND OR NOT ( ) ATOM \\EOF') ==\
            ['\AND', '\OR', '\NOT', '\ATOM', '\EOF']
        
        self.index.index('AND OR EOF NOT')
        self.index._index.initSearch()
        assert self.index.search('AND OR EOF NOT')[0] == 1

    def test_keywords_title(self) :
        assert self.index._parseTerms(u'`And Or Not ( ) Atom \\Eof') ==\
            ['\AND', '\OR', '\NOT', '\ATOM', '\EOF']
        
        self.index.index('And Or Eof Not')
        self.index._index.initSearch()
        assert self.index.search('And Or Eof Not')[0] == 1


    def test_empty_search(self) :
        self.index._index.initSearch()
        assert self.index.search('') == []

    def test_wildcards(self) :
        self.index.index('f\o')
        self.index.index('f*')
        self.index._index.initSearch()
        assert len(self.index.search('f*')) == 1
        assert self.index._parseTerms('f*o') == ['f*o']

if __name__ == "__main__":
    unittest.main()
