#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    best_score = 0
    best_alpha = 0.01

    alpha_tester = AlphaTester(original_data_model, trainer)

    for alpha in search_space:

        score_jobs = [pool.apply_async(alpha_tester, 
                                       (training, validation, alpha))
                      for training, validation in 
                      kFolds(training_data, k)]

        scores = [job.get() for job in score_jobs]
        
        average_score = reduceScores(scores)

        logger.debug("Average Score: %f, alpha: %s" % (average_score, alpha))

        if average_score >= best_score :
            best_score = average_score
            best_alpha = alpha

    logger.info('optimum alpha: %f' % best_alpha)
    pool.close()
    pool.join()


    return best_alpha

# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
def kFolds(training_data, k):
    if k < 2 :
        raise ValueError("Number of folds must be at least 2")
    
    if len(training_data) < 2 :
        raise ValueError("At least two traning datum are required")

    train_dtype = training_data.dtype
    slices = [training_data[i::k] for i in xrange(k)]
    for i in xrange(k):
        validation = slices[i]
        training = [datum for s in slices if s is not validation for datum in s]
        validation = numpy.array(validation, train_dtype)
        training = numpy.array(training, train_dtype)

        if len(training) and len(validation) :
            yield (training, validation)
        else :
            warnings.warn("Only providing %s folds out of %s requested" % 
                          (i, k))
            break

class AlphaTester(object) :
    def __init__(self, data_model, trainer) : # pragma : no cover
        self.data_model = data_model
        self.trainer = trainer

    def __call__(self, training, validation, alpha) :
        data_model = self.trainer(training, self.data_model, alpha)

        weight = numpy.array([field.weight
                              for field in 
                              data_model['fields']])
        bias = data_model['bias']

        predictions = numpy.dot(validation['distances'], weight) + bias
        true_labels = validation['label'] == 'match'

        return scorePredictions(true_labels, predictions)
        
def scorePredictions(true_labels, predictions) :

    true_dupes = numpy.sum(true_labels)
    true_predicted_dupes = numpy.sum(predictions[true_labels == 1] > 0)

    if not true_dupes :
        score = None

    elif true_predicted_dupes :

        predicted_dupes = numpy.sum(predictions > 0)
        true_dupes = numpy.sum(true_labels)

        recall = true_predicted_dupes/float(true_dupes)
        precision = true_predicted_dupes/float(predicted_dupes)

        score = 2 * recall * precision / (recall + precision)

    else :
        score = 0

    return score

def reduceScores(scores) :
    
    scores = [score for score in scores if score is not None]

    if scores :
        average_score = sum(scores)/float(len(scores))
    else :
        average_score = 0

    return average_score


