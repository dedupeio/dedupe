#!/usr/bin/python
# -*- coding: utf-8 -*-
import core
from random import shuffle
import copy
import numpy


# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
def gridSearch(training_data,
               trainer,
               original_data_model,
               k=10,
               search_space=[.0001, .001, .01, .1, 1],
               randomize=True,
               ):

    numpy.random.shuffle(training_data)

    print 'using cross validation to find optimum alpha...'
    scores = []

    fields = sorted(original_data_model['fields'].keys())

    for alpha in search_space:
        all_score = 0
        all_N = 0
        for (training, validation) in kFolds(training_data, k):
            data_model = trainer(training, original_data_model, alpha)

            weight = numpy.array([data_model['fields'][field]['weight']
                                 for field in fields])
            bias = data_model['bias']

            real_labels = training_data['label']
            valid_examples = training_data['field_distances']
            valid_scores = numpy.dot(valid_examples, weight) + bias


            predicted_labels = valid_scores > 0


            score = numpy.sum(real_labels == predicted_labels)
            


            all_score += score
            all_N += len(real_labels)

        #print alpha, float(all_score) / all_N
        scores.append(float(all_score) / all_N)

    best_alpha = search_space[::-1][scores[::-1].index(max(scores))]

    print 'optimum alpha: ', best_alpha
    return best_alpha


def kFolds(training_data, k):
    train_dtype = training_data.dtype
    slices = [training_data[i::k] for i in xrange(k)]
    for i in xrange(k):
        validation = slices[i]
        training = [datum for s in slices if s is not validation
                    for datum in s]
        validation = numpy.array(validation, train_dtype)
        training = numpy.array(training, train_dtype)

        yield (training, validation)
