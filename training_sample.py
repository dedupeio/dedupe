# provides functions for selecting a sample of training data
from random import sample, shuffle
from itertools import combinations
from dedupe import *

# create a set of training data
def trainingDistances(training_pairs, data_model) :
  training_data = []

  for label, examples in training_pairs.items() :
      for pair in examples :
          distances = calculateDistance(pair[0],
                                        pair[1],
                                        data_model['fields'])
          training_data.append((label, distances))

  shuffle(training_data)
  return training_data

# create a random set of training pairs based on known duplicates
def randomTrainingPairs(data_d,
                        duplicates_s,
                        n_training_dupes,
                        n_training_distinct) :
  duplicates = []

  duplicates_set = list(duplicates_s)
  shuffle(duplicates_set)


  for random_pair in duplicates_set :
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])      
    duplicates.append(training_pair)
    if len(duplicates) == n_training_dupes :
      break

  nonduplicates = []

  all_pairs = list(combinations(data_d, 2))

  for random_pair in sample(all_pairs,
                            n_training_dupes + n_training_distinct) :
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])
    if set(random_pair) not in duplicates_s :
      nonduplicates.append(training_pair)
    if len(nonduplicates) == n_training_distinct :
      break
      
  return({0:nonduplicates, 1:duplicates})
  
## user training functions ##

# based on the data model and training we have so far, returns the n record pairs we are least certain about
def findUncertainPairs(record_distances, data_model, num_pairs) :

  duplicateScores = scorePairs(record_distances, data_model)

  uncertain_pairs = []
  for scored_pair in duplicateScores :
    pair = scored_pair.keys()[0]
    dict_pair = (data_d[pair[0]], data_d[pair[1]])
    probability = scored_pair.values()[0]

    entropy = -(probability * log(probability, 2) +
                  (1-probability) * log(1-probability, 2))
    
    uncertain_pairs.append({dict_pair : entropy})

    
  uncertain_pairs = sorted(uncertain_pairs, key=lambda(d): -d.values()[0])
  return uncertain_pairs[0:num_pairs]

# loop for user to enter training data
def activeLearning(data_d, data_model, labelPairFunction) :
  training_data = []
  pairs = allCandidates(data_d) 
  record_distances = recordDistances(pairs, data_d, data_model)
  for _ in range(2) :
    print "finding the next uncertain pair ..."
    uncertain_pairs = findUncertainPairs(record_distances, data_model, 1)
    labeled_pairs = labelPairFunction(uncertain_pairs)
    training_data = addTrainingData(labeled_pairs, training_data)
    data_model = trainModel(training_data, numIterations, data_model)
  
  return(training_data, data_model)

def fischerInformation(data_model) :
  fic_score = 0
  
  return fic_score

# appends training data to the training data collection  
def addTrainingData(labeled_pairs, training_data) :

  for label, examples in labeled_pairs.items() :
      for pair in examples :
          distances = calculateDistance(pair[0],
                                        pair[1],
                                        data_model['fields'])
          training_data.append((label, distances))
          
  return training_data
  
if __name__ == '__main__':
  from canonical_example import init

  # user defined function to label pairs as duplicates or non-duplicates
  def consoleLabel(uncertain_pairs) :
    duplicates = []
    nonduplicates = []
    
    for pair in uncertain_pairs :
      label = ''
      
      record_pair = pair.keys()[0]
      for instance in record_pair :
        print instance.values()
      
      print "Do these records refer to the same thing?"  
      
      valid_response = False
      while not valid_response :
        label = raw_input('(y)es / (n)o / (u)nsure\n')
        if label in ['y', 'n', 'u'] :
          valid_response = True

      if label == 'y' :
        duplicates.append(record_pair)
      elif label == 'n' :
        nonduplicates.append(record_pair)
      elif label != 'u' :
        print 'Nonvalid response'
        raise

    return({0:nonduplicates, 1:duplicates})

  
  num_training_dupes = 200
  num_training_distinct = 16000
  numIterations = 100

  import time
  t0 = time.time()
  (data_d, duplicates_s, data_model) = init()
  #candidates = allCandidates(data_d)
  #print "training data: "
  #print duplicates_s

  print "number of duplicates pairs"
  print len(duplicates_s)
  print ""

  #lets do some active learning here
  #training_pairs = activeLearning(data_d, data_model, consoleLabel);
  
  #profiling
  
  labelPairFunction = consoleLabel
  training_data = []
  pairs = allCandidates(data_d) 
  record_distances = recordDistances(pairs, data_d, data_model)
  for _ in range(2) :
    print "finding the next uncertain pair ..."
    uncertain_pairs = findUncertainPairs(record_distances, data_model, 1)
    labeled_pairs = labelPairFunction(uncertain_pairs)
    training_data = addTrainingData(labeled_pairs, training_data)
    data_model = trainModel(training_data, numIterations, data_model)
