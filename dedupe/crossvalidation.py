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
               k=3,
               search_space=[.00001, .0001, .001, .01, .1, 1],
               randomize=True):

    training_data = training_data[numpy.random.permutation(training_data.size)]

    logger.info('using cross validation to find optimum alpha...')
    scores = []

    fields = original_data_model['fields']

    for alpha in search_space:
        all_score = 0
        all_N = 0
        for (training, validation) in kFolds(training_data, k):
            data_model = trainer(training, original_data_model, alpha)

            weight = numpy.array([field.weight
                                  for field in fields])
            bias = data_model['bias']

            labels = validation['label'] == 'match'
            predictions = numpy.dot(validation['distances'], weight) + bias

            true_dupes = numpy.sum(labels == 1)

            if true_dupes == 0 :
                logger.warning("not real positives, change size of folds")
                continue

            true_predicted_dupes = numpy.sum(predictions[labels == 1] > 0)

            recall = true_predicted_dupes/float(true_dupes)

            if recall == 0 :
                score = 0

            else:
                precision = true_predicted_dupes/float(numpy.sum(predictions > 0))
                score = 2 * recall * precision / (recall + precision)


            all_score += score

        average_score = all_score/k
        logger.debug("Average Score: %f", average_score)

        scores.append(average_score)

    best_alpha = search_space[::-1][scores[::-1].index(max(scores))]

    logger.info('optimum alpha: %f' % best_alpha)
    return best_alpha


def kFolds(training_data, k):
    train_dtype = training_data.dtype
    slices = [training_data[i::k] for i in xrange(k)]
    for i in xrange(k):
        validation = slices[i]
        training = [datum for s in slices if s is not validation for datum in s]
        validation = numpy.array(validation, train_dtype)
        training = numpy.array(training, train_dtype)

        yield (training, validation)
