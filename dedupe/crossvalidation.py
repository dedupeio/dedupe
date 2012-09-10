import core
from random import shuffle
import copy
import numpy

#http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/

def gridSearch(training_data,
               trainer,
               original_data_model,
               k = 10,
               search_space = [.0001, .001, .01, .1, 1],
               randomize=True) :



  numpy.random.shuffle(training_data)

  print "using cross validation to find optimum alpha"
  scores = []

  fields = training_data[0][1][0]

  for alpha in search_space :
    all_score = 0
    all_N = 0
    for training, validation in kFolds(training_data, k) :
      data_model = trainer(training, original_data_model, alpha)
      
      weight = numpy.array([data_model['fields'][field]['weight'] for field in fields]) 

      (real_labels,
       validation_distances) = zip(*[(label, distances)
                                     for label, distances in validation])

      predicted_labels = []
      bias = data_model["bias"]
      for example in validation_distances :
        prediction = bias + numpy.dot(weight, example[1])
        if prediction > 0 :
          predicted_labels.append(1)
        else :
          predicted_labels.append(0)



      score = 0
      for real_label, predicted_label in zip(real_labels, predicted_labels) :
        if real_label == predicted_label :
          score += 1

      all_score += score
      all_N += len(real_labels)

    print alpha, float(all_score)/all_N
    scores.append(float(all_score)/all_N)

  best_alpha = search_space[::-1][scores[::-1].index(max(scores))]
  
  return best_alpha

def kFolds(training_data, k):
    train_dtype = training_data.dtype
    slices = [training_data[i::k] for i in xrange(k)]
    for i in xrange(k):
        validation = slices[i]
        training = [datum 
                    for s in slices if s is not validation
                    for datum in s]
        validation = numpy.array(validation, train_dtype)
        training = numpy.array(training, train_dtype)

        yield training, validation

