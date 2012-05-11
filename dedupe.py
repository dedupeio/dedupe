import itertools
import math
#import distance #libdistance library http://monkey.org/~jose/software/libdistance/
import affinegap
import pegasos


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
        print (data_d[pair[0]],data_d[pair[1]])
        print score
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
  train_pairs = {}
  while len(train_pairs) < n :
    random_pair = frozenset(random.sample(data_d.keys(), 2))
    if random_pair in duplicates_s : 
      train_pairs[random_pair] = 1
    else :
      train_pairs[random_pair] = 0
      
  return(train_pairs)

def createTrainingData(data_d, duplicates_s, n, data_model) :
  train_pairs = createTrainingPairs(data_d, duplicates_s, n)

  training_data = []
  for pair in train_pairs :
      instance_1 = data_d[tuple(pair)[0]]
      instance_2 = data_d[tuple(pair)[1]]
      distances = calculateDistance(instance_1,
                                    instance_2,
                                    data_model['fields'])
      training_data.append((train_pairs[pair], distances))

  return (training_data, train_pairs)

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

    
def trainBlocking(data_d, train_pairs, data_model, predicates) :

  eta = len(train_pairs)
  
  knownGoodPairs = []
  knownBadPairs = []
  
  for pair in train_pairs :
    if train_pairs[pair] == 1 :
      knownGoodPairs.append(tuple(pair))
    else :
      knownBadPairs.append(tuple(pair))
  
  foundGoodPairs = {}
  foundBadPairs = {}
  
  numGoodPairs = 0
  numBadPairs = 0
  
  for fieldName in data_model['fields'] :
    for pair in train_pairs :
      label = train_pairs[pair]
      instance_1 = data_d[tuple(pair)[0]]
      instance_2 = data_d[tuple(pair)[1]]
      
      for predicate in predicates :
        if predicate(instance_1[fieldName]) == predicate(instance_2[fieldName]) :
          numGoodPairs += 1
          #clean this up - there's a better way w/o try/except
          try :
            foundGoodPairs[(predicate,fieldName)] += 1
          except KeyError :
            foundGoodPairs[(predicate,fieldName)] = 1
        else :
          numBadPairs += 1
          try :
            foundBadPairs[(predicate,fieldName)] += 1
          except KeyError :
            foundBadPairs[(predicate,fieldName)] = 1
  
  print "foundGoodPairs: "
  print foundGoodPairs
  print "foundBadPairs: "
  print foundBadPairs  
  
  predicateSet = set(foundGoodPairs.keys()).union(set(foundBadPairs.keys()))
  print "predicateSet: "
  print predicateSet
  
  filteredPredicateSet = set()
  for predicate in predicateSet :
    if foundBadPairs[predicate] < eta :
      filteredPredicateSet.add(predicate)
      
  print "filteredPredicateSet: "
  print filteredPredicateSet
  
  expectedBadPairs = math.sqrt(len(train_pairs) / math.log(numGoodPairs))
  print "expectedBadPairs: ", expectedBadPairs
  
      
  filteredBadPairs = knownBadPairs[:]   
  predicateCount = {}
  for pair in knownBadPairs :
    try :
      predicateCount[pair] += 1
    except KeyError :
      predicateCount[pair] = 1
   
    if predicateCount[pair] > expectedBadPairs :
      filteredBadPairs.remove(pair)
  
  print "numGoodPairs: ", numGoodPairs
  print "numBadPairs: ", numBadPairs
      
  print "filteredBadPairs: "
  print len(filteredBadPairs)
  
  #print "knownBadPairs: "
  #print knownBadPairs
  
  filteredBadCoverage, numCoveredPairs = predicateDegree(data_d, data_model, filteredBadPairs, predicates)
  
  print "filteredBadCoverage: "
  print filteredBadCoverage
  
  print "numCoveredPairs: ", numCoveredPairs
  
  epsilon = 1
  finalPredicateSet = []
  consideredPredicates = filteredBadCoverage
  while numGoodPairs >= epsilon :
    optimumCover = 0
    foundCoverage = 0
    bestPredicate = None
    for predicate in consideredPredicates :
      cover = foundGoodPairs[predicate] / float(filteredBadCoverage[predicate])
      if cover > optimumCover :
        optimumCover = cover
        foundCoverage = foundGoodPairs[predicate]
        bestPredicate = predicate

    if not bestPredicate : break

    consideredPredicates.pop(bestPredicate)
    numGoodPairs = numGoodPairs - foundCoverage
    finalPredicateSet.append(bestPredicate)
    
  print "FINAL PREDICATE SET!!!!"
  print finalPredicateSet
  return finalPredicateSet

def predicateDegree(data_d, data_model, pairs, predicates) :
  numPairs = 0
  foundPairs = {}
  
  for fieldName in data_model['fields'] :
    for pair in pairs :
      instance_1 = data_d[tuple(pair)[0]]
      instance_2 = data_d[tuple(pair)[1]]
    
      for predicate in predicates :
        if predicate(instance_1[fieldName]) == predicate(instance_2[fieldName]) :
          numPairs += 1
          #clean this up - there's a better way w/o try/except
          try :
            foundPairs[(predicate,fieldName)] += 1
          except KeyError :
            foundPairs[(predicate,fieldName)] = 1
  
  return (foundPairs, numPairs)



#returns the field as a tuple
def wholeFieldPredicate(field) :

  return (field, )
  
#returns the tokens in the field as a tuple, split on whitespace
def tokenFieldPredicate(field) :
  
  return field.split()
  
    
def run(numTrainingPairs, numIterations) :
  import time
  t0 = time.time()
  data_d, header, duplicates_s = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print duplicates_s
  
  print "number of known duplicates: "
  print len(duplicates_s)

  training_data, train_pairs = createTrainingData(data_d, duplicates_s, numTrainingPairs, data_model)
  #print "training data from known duplicates: "
  #print training_data
  print "number of training items: "
  print len(training_data)

  return(training_data)
  ##data_model = trainModel(training_data, numIterations, data_model)
  
  ## print "finding duplicates ..."
  ## dupes = findDuplicates(candidates, data_d, data_model, -.5)
  ## true_positives = 0
  ## false_positives = 0
  ## for dupe_pair in dupes :
  ##   if set(dupe_pair.keys()[0]) in duplicates_s :
  ##       true_positives += 1
  ##   else :
  ##       false_positives += 1

  ## print "precision"
  ## print (len(dupes) - false_positives)/float(len(dupes))

  ## print "recall"
  ## print true_positives/float(len(duplicates_s))
  ## print "ran in ", time.time() - t0, "seconds"

  ## print data_model

if __name__ == '__main__':
  td = run(8000,300)
