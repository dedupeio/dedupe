from itertools import combinations
from random import sample
import affinegap
import lr
from blocking import trainBlocking
from predicates import *


def createRandomTrainingPairs(data_d, duplicates_s, n) :
  duplicates = []
  nonduplicates = []
  random_pairs = sample(list(combinations(data_d, 2)), n)
  for random_pair in random_pairs :
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])
    if set(random_pair) in duplicates_s :
      duplicates.append(training_pair)
    else:
      nonduplicates.append(training_pair)

      
  return(nonduplicates, duplicates)

def createOverSampleTrainingPairs(data_d, duplicates_s, n) :
  duplicates = []

  for random_pair in duplicates_s :
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])      
    duplicates.append(training_pair)

  n -= len(duplicates)
  nonduplicates = []

  all_pairs = list(combinations(data_d, 2))

  for random_pair in sample(all_pairs, n) :
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])
    if set(random_pair) not in duplicates_s :
      nonduplicates.append(training_pair)
      
  return(nonduplicates, duplicates)

def calculateDistance(instance_1, instance_2, fields) :
  distances_d = {}
  for name in fields :
    if fields[name]['type'] == 'String' :
      distanceFunc = affinegap.normalizedAffineGapDistance
    distances_d[name] = distanceFunc(instance_1[name],instance_2[name])

  return distances_d

def createTrainingData(training_pairs) :
  training_data = []

  # Use 0,1 labels for logistic regression -1,1 for SVM
  nonduplicate_label = 0
  duplicate_label = 1
  training_pairs = zip((nonduplicate_label,
                        duplicate_label),
                       training_pairs)
                            
  for label, examples in training_pairs :
      for pair in examples :
          distances = calculateDistance(pair[0],
                                        pair[1],
                                        data_model['fields'])
          training_data.append((label, distances))

  return training_data

def trainModel(training_data, iterations, data_model) :
    trainer = lr.LogisticRegression()
    trainer.train(training_data, iterations)

    data_model['bias'] = trainer.bias
    for name in data_model['fields'] :
        data_model['fields'][name]['weight'] = trainer.weight[name]

    return(data_model)


def identifyCandidates(data_d) :
  return [data_d.keys()]

def findDuplicates(candidates, data_d, data_model, threshold) :
  duplicateScores = []

  for candidates_set in candidates :
    for pair in combinations(candidates_set, 2):
      fields = data_model['fields']
      distances = calculateDistance(data_d[pair[0]], data_d[pair[1]], fields)

      score = data_model['bias'] 
      for name in fields :
        score += distances[name] * fields[name]['weight']

      #print (pair, score)
      if score > threshold :
        #print (data_d[pair[0]],data_d[pair[1]])
        #print score
        duplicateScores.append({ pair : score })
  
  return duplicateScores

if __name__ == '__main__':
  from test_data import init
  numTrainingPairs = 16000
  numIterations = 20

  import time
  t0 = time.time()
  (data_d, duplicates_s, data_model) = init()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print duplicates_s
  
  print "number of known duplicates: "
  print len(duplicates_s)

  training_pairs = createOverSampleTrainingPairs(data_d, duplicates_s, numTrainingPairs)
  #training_pairs = createRandomTrainingPairs(data_d, duplicates_s, numTrainingPairs)

  trainBlocking(training_pairs,
                (wholeFieldPredicate,
                 tokenFieldPredicate,
                 commonIntegerPredicate,
                 sameThreeCharStartPredicate,
                 sameFiveCharStartPredicate,
                 sameSevenCharStartPredicate,
                 nearIntegersPredicate,
                 commonFourGram,
                 commonSixGram),
                data_model, 1, 1)  
  
  training_data = createTrainingData(training_pairs)
  #print "training data from known duplicates: "
  #print training_data
  print "number of training items: "
  print len(training_data)

  data_model = trainModel(training_data, numIterations, data_model)

  print data_model
  print "finding duplicates ..."
  dupes = findDuplicates(candidates, data_d, data_model, -3)
  dupe_ids = set([frozenset(list(dupe_pair.keys()[0])) for dupe_pair in dupes])
  true_positives = dupe_ids & duplicates_s
  false_positives = dupe_ids - duplicates_s
  uncovered_dupes = duplicates_s - dupe_ids

  print "precision"
  print (len(dupes) - len(false_positives))/float(len(dupes))

  print "recall"
  print  len(true_positives)/float(len(duplicates_s))
  print "ran in ", time.time() - t0, "seconds"

  print data_model

  for pair in uncovered_dupes :
         print ""
         print (data_d[tuple(pair)[0]], data_d[tuple(pair)[1]])



  

