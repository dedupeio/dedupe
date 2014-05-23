from collections import defaultdict
import math
import numpy

class CosineSimilarity :
    """
    Defines a class version of the closure. The pure closure
    version is slightly faster but can't be saved (pickled) in settings file.

    Terms with frequencies greater than n_documents * idf_threshold are discarded.
    """

    def __init__(self, corpus):
        if corpus :

            self.doc_freq = defaultdict(float)
            num_docs = 0.0
            for document in corpus :
                for word in set(document.split()) :
                    self.doc_freq[word] += 1
                num_docs += 1

            for word, count in self.doc_freq.items() :
                self.doc_freq[word] = math.log(num_docs/count)

        else :
            self.doc_freq = defaultdict(one)

        self.vectors = {}

    def vectorize(self, string) :
        if string in self.vectors :
            return self.vectors[string]

        vector = {}
        for word in string.split() :
            if word in vector :
                vector[word] += self.doc_freq[word]
            else :
                vector[word] = self.doc_freq[word]

        norm = math.sqrt(sum(weight * weight for weight in vector.values()))

        self.vectors[string] = (vector, norm)

        return vector, norm


    def __call__(self, string_1, string_2):
        vector_1, norm_1 = self.vectorize(string_1)
        vector_2, norm_2 = self.vectorize(string_2)

        if norm_1 and norm_2 :
            numerator = 0.0
            for word in set(vector_1) & set(vector_2) :
                numerator += vector_1[word] * vector_2[word]

            return numerator/(norm_1 * norm_2)

        else :
            return numpy.nan


def one() :
    return 1.0
        

if __name__ == '__main__' :

    corpus = ['tom foolery', 'bob sillyery', 'joan']
    
    sim = CosineSimilarity(corpus, 1)
    
    print sim('foolery', 'joan foolery')
    print sim('foolery', 'joan foolery')




