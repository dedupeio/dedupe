import itertools
import math
#import distance #libdistance library http://monkey.org/~jose/software/libdistance/
import affinegap
import lr
#import pegasos
from collections import defaultdict


def hashPair(pair) :
      return tuple(sorted([tuple(sorted(pair[0].items())), tuple(sorted(pair[1].items()))]))

def canonicalImport(filename) :
    import csv

    data_d = {}
    duplicates_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        print header
        for i, row in enumerate(reader) :
            instance = {}
            for j, col in enumerate(row) :
              if header[j] == 'unique_id' :
                duplicates_d.setdefault(col, []).append(i)
              else :
                instance[header[j]] = col.strip().strip('"').strip("'")
                
            data_d[i] = instance

    duplicates_s = set([])
    for unique_id in duplicates_d :
      if len(duplicates_d[unique_id]) > 1 :
        for pair in itertools.combinations(duplicates_d[unique_id], 2) :
          duplicates_s.add(frozenset(pair))

    return(data_d, header, duplicates_s)

def dataModel() :
  return  {'fields': 
            {'name' : {'type': 'String', 'weight' : 1}, 
             'address' : {'type' :'String', 'weight' : 1},
             'city' : {'type': 'String', 'weight' : 1},
             'cuisine' : {'type': 'String', 'weight' : 1}
            },
           'bias' : 0}

def identifyCandidates(data_d) :
  return [data_d.keys()]

def findDuplicates(candidates, data_d, data_model, threshold) :
  duplicateScores = []

  for candidates_set in candidates :
    for pair in itertools.combinations(candidates_set, 2):
      scorePair = {}
      score = data_model['bias'] 
      fields = data_model['fields']

      distances = calculateDistance(data_d[pair[0]], data_d[pair[1]], fields)
      for name in fields :
        score += distances[name] * fields[name]['weight']
      scorePair[pair] = score
      #print (pair, score)
      if score > threshold :
        #print (data_d[pair[0]],data_d[pair[1]])
        #print score
        duplicateScores.append(scorePair)
  
  return duplicateScores

def calculateDistance(instance_1, instance_2, fields) :
  distances_d = {}
  for name in fields :
    if fields[name]['type'] == 'String' :
      distanceFunc = affinegap.normalizedAffineGapDistance
    x = distanceFunc(instance_1[name],instance_2[name])
    distances_d[name] = x

  return distances_d

def createTrainingPairs(data_d, duplicates_s, n) :
  import random
  nonduplicates_s = set([])
  duplicates = []
  nonduplicates = []
  nPairs = 0
  while nPairs < n :
    random_pair = frozenset(random.sample(data_d, 2))
    training_pair = (data_d[tuple(random_pair)[0]],
                     data_d[tuple(random_pair)[1]])
    if random_pair in duplicates_s :
      duplicates.append(training_pair)
      nPairs += 1
    elif random_pair not in nonduplicates_s :
      nonduplicates.append(training_pair)
      nonduplicates_s.add(random_pair)
      nPairs += 1
      
  return(nonduplicates, duplicates)

def createTrainingData(training_pairs) :

  training_data = []

  for label, examples in enumerate(training_pairs) :
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

def trainModelSVM(training_data, iterations, data_model) :

    labels, vectors = zip(*training_data)

    keys = data_model['fields'].keys()
    vectors = [[_[key] for key in keys] for _ in vectors]
    
    trainer = pegasos.PEGASOS()

    trainer.train((labels, vectors))

    data_model['bias'] = trainer.bias
    for i, name in enumerate(keys) :
        data_model['fields'][name]['weight'] = trainer.lw[i]

    return(data_model)

def predicateCoverage(pairs, predicates) :
    coverage = defaultdict(list)
    for pair in pairs :
        for predicate, field in predicates :
            keys1 = predicate(pair[0][field])
            keys2 = predicate(pair[1][field])
            if set(keys1) & set(keys2) :
                coverage[(predicate,field)].append(pair)
              
    return(coverage)


# Approximate learning of blocking following the ApproxRBSetCover from
# page 102 of Bilenko
def trainBlocking(training_pairs, predicates, data_model, eta, epsilon) :

  training_distinct, training_dupes = training_pairs
  n_training_dupes = len(training_dupes)
  n_training_distinct = len(training_distinct)
  sample_size = n_training_dupes + n_training_distinct

  # The set of all predicate functions operating over all fields
  predicateSet = list(itertools.product(predicates, data_model['fields']))
  n_predicates = len(predicateSet)

  
  found_dupes = predicateCoverage(training_dupes,
                                  predicateSet)
  found_distinct = predicateCoverage(training_distinct,
                                     predicateSet)


  predicateSet = found_dupes.keys() 

  # We want to throw away the predicates that puts together too many
  # distinct pairs
  eta = sample_size * eta

  [predicateSet.remove(predicate)
   for predicate in found_distinct
   if len(found_distinct[predicate]) >= eta]

  # We don't want to penalize a blocker if it puts distinct pairs
  # together that look like they could be duplicates. Here we compute
  # the expected number of predicates that will cover a duplicate pair
  # We'll remove all the distince pairs from consideration if they are
  # covered by many predicates
  expected_dupe_cover = math.sqrt(n_predicates / math.log(n_training_dupes))

  predicate_count = defaultdict(int)
  for pair in itertools.chain(*found_distinct.values()) :
      predicate_count[hashPair(pair)] += 1

  training_distinct = [pair for pair in training_distinct
                       if predicate_count[hashPair(pair)] < expected_dupe_cover]


  found_distinct = predicateCoverage(training_distinct,
                                     predicateSet)

  # Greedily find the predicates that, at each step, covers the most
  # duplicates and covers the least distinct pairs, dute to Chvatal, 1979
  finalPredicateSet = []
  print "Uncovered dupes"
  print n_training_dupes
  while n_training_dupes >= epsilon :

    optimumCover = 0
    bestPredicate = None
    for predicate in predicateSet :
      try:  
          cover = (len(found_dupes[predicate])
                   / float(len(found_distinct[predicate]))
                   )
      except ZeroDivisionError:
          cover = len(found_dupes[predicate])

      if cover > optimumCover :
        optimumCover = cover
        bestPredicate = predicate


    if not bestPredicate :
        print "Ran out of predicates"
        break

    predicateSet.remove(bestPredicate)
    n_training_dupes -= len(found_dupes[bestPredicate])
    [training_dupes.remove(pair) for pair in found_dupes[bestPredicate]]
    found_dupes = predicateCoverage(training_dupes,
                                    predicateSet)

    print n_training_dupes

    finalPredicateSet.append(bestPredicate)
    
  print "FINAL PREDICATE SET!!!!"
  print finalPredicateSet

  return finalPredicateSet





#returns the field as a tuple
def wholeFieldPredicate(field) :

  return (field, )
  
#returns the tokens in the field as a tuple, split on whitespace
def tokenFieldPredicate(field) :
  
  return field.split()


# Contain common integer
def commonIntegerPredicate(field) :
    import re
    return re.findall("\d+", field)

def nearIntegersPredicate(field) :
    import re
    ints = sorted([int(i) for i in re.findall("\d+", field)])
    return [(i-1, i, i+1) for i in ints]

def commonFourGram(field) :
    return (field[pos:pos + 4] for pos in xrange(0, len(field), 4))

def commonSixGram(field) :
    return (field[pos:pos + 6] for pos in xrange(0, len(field), 6))

def sameThreeCharStartPredicate(field) :
    return field[:2]

def sameFiveCharStartPredicate(field) :
    return field[:4]

def sameSevenCharStartPredicate(field) :
    return field[:6]




if __name__ == '__main__':
  numTrainingPairs = 16000
  numIterations = 50

  import time
  t0 = time.time()
  data_d, header, duplicates_s = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print duplicates_s
  
  print "number of known duplicates: "
  print len(duplicates_s)

  training_pairs = createTrainingPairs(data_d, duplicates_s, numTrainingPairs)

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
  
  print "finding duplicates ..."
  dupes = findDuplicates(candidates, data_d, data_model, -2)
  true_positives = 0
  false_positives = 0
  for dupe_pair in dupes :
    if set(dupe_pair.keys()[0]) in duplicates_s :
        true_positives += 1
    else :
        false_positives += 1

  print "precision"
  print (len(dupes) - false_positives)/float(len(dupes))

  print "recall"
  print true_positives/float(len(duplicates_s))
  print "ran in ", time.time() - t0, "seconds"

  print data_model

  

