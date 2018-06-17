import numpy
import binascii

class MinHasher(object):
    def __init__(self, n_hashes):
        max_hash = (2**64)/2 - 1
        
        self.a = numpy.random.randint(-max_hash, max_hash, size=(n_hashes, 1))
        self.b = numpy.random.randint(-max_hash, max_hash, size=(n_hashes, 1))

    #@profile
    def __call__(self, values):
        X = numpy.array([hash(value) for value in values])
        # http://www.iip.ist.i.kyoto-u.ac.jp/informatics-seminar/lib/exe/fetch.php?media=bottomk-talk.pdf
        #
        # not this is not completely correct
        h = self.a * X
        h += self.b
        h >>= 32
        mins = numpy.amin(h, axis=1)
        #mins = mins.astype('str')
        #mins = set(mins)
        return mins
    
if __name__ == '__main__':
    def bar():
        n = 36
        X = []
        A = set(str(x) for x in range(10))
        B = set(str(x) for x in range(7))
        for _ in range(10000):
            mh = MinHasher(n)
            
            a = mh(A)
            b = mh(B)
    
            x = len(a & b)/n
            X.append(x)
        print(numpy.mean(X))
        print(numpy.std(X))
          
    
      
    bar()
