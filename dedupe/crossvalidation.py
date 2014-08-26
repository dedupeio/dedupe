#!/usr/bin/python
# -*- coding: utf-8 -*-

import core
from random import shuffle
import copy
import numpy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/

def gridSearch(training_data,
               trainer,
               original_data_model,
               num_cores,
               k=3,
               search_space=[.00001, .0001, .001, .01, .1, 1],
               randomize=True):

    if num_cores < 2 :
        from multiprocessing.dummy import Pool
    else :
        from backport import Pool

    pool = Pool()

    training_data = training_data[numpy.random.permutation(training_data.size)]

    logger.info('using cross validation to find optimum alpha...')
    scores = []

    fields = original_data_model['fields']

    for alpha in search_space:

        fold_data = kFolds(training_data,k)

        fold_scores = [pool.apply_async(trainAndScore,(alpha,original_data_model,trainer)+fd)
                       for fd in fold_data]
        
        all_score = sum([fs.get() for fs in fold_scores])
        
        average_score = all_score/k
        logger.debug("Average Score: %f", average_score)

        scores.append(average_score)

    best_alpha = search_space[::-1][scores[::-1].index(max(scores))]

    logger.info('optimum alpha: %f' % best_alpha)
    pool.close()
    pool.join()


    return best_alpha



def trainAndScore(alpha, data_model, trainer, training, validation):
    data_model = trainer(training, data_model, alpha)

    weight = numpy.array([field.weight
                          for field in data_model['fields']])
    bias = data_model['bias']

    labels = validation['label'] == 'match'
    predictions = numpy.dot(validation['distances'], weight) + bias

    true_dupes = numpy.sum(labels == 1)

    if true_dupes == 0 :
        logger.warning("not real positives, change size of folds")
        return 

    true_predicted_dupes = numpy.sum(predictions[labels == 1] > 0)
            
    recall = true_predicted_dupes/float(true_dupes)

    if recall == 0 :
        score = 0

    else:
        precision = true_predicted_dupes/float(numpy.sum(predictions > 0))
        score = 2 * recall * precision / (recall + precision)

    return score



def kFolds(training_data, k):
    train_dtype = training_data.dtype
    slices = [training_data[i::k] for i in xrange(k)]
    for i in xrange(k):
        validation = slices[i]
        training = [datum for s in slices if s is not validation for datum in s]
        validation = numpy.array(validation, train_dtype)
        training = numpy.array(training, train_dtype)

        yield (training, validation)
