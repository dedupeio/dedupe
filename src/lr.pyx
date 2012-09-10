#cython: boundscheck=False, wraparound=False, cdivision=True

from libc cimport math
cimport numpy


def lr(numpy.ndarray[numpy.int32_t, ndim=1] labels,
       numpy.ndarray[numpy.float32_t, ndim=2] examples,
       float alpha) :

    cdef int i, j, n, label, n_features, n_examples
    cdef float rate, rate_n, bias, predicted, update, error, logit

    cdef numpy.ndarray[numpy.float32_t, ndim=1] weight

    n_examples = examples.shape[0]
    n_features = examples.shape[1]

    n = 500
    rate = 0.01

    weight = examples[0] * 0
    bias = 0

    for i in range(n):
        rate_n = rate * (n-i)/n
        for j in range(n_examples) :
            label = labels[j]

            logit = bias
            for k in range(n_features) :
                logit += weight[k] * examples[j,k]

            predicted = 1.0 / (1.0 + math.exp(-logit))
                
            error = label - predicted
            
            for k in range(n_features) :
                update = error * examples[j,k] - (alpha * weight[k])
                weight[k] += rate_n * update
            
            bias += rate_n * error
        #print 'iteration', i, 'done.'
    return weight, bias 

