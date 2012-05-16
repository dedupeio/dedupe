import itertools
import math
#import distance #libdistance library http://monkey.org/~jose/software/libdistance/
import affinegap
import lr
#import pegasos


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

    
def trainBlocking(training_pairs, predicates, fields) :

# Much of this would be easier if the pair functions were hashable
# Maybe something like tuple(sorted(d.iteritems())) then we could leverage the set operations

  eta = len(training_pairs)
  
  knownGoodPairs, knownBadPairs = training_pairs
  
  def predicateCoverage(pairs, predicates, fields) :
      coverage = {}
      covered_pairs = 0
      for pair in pairs :
          covered = False
          for field in fields :
              for predicate in predicates :
                  keys1 = predicate(pair[0][field])
                  keys2 = predicate(pair[1][field])
                  if set(keys1) & set(keys2) :
                      coverage.setdefault((predicate,field),[]).append(pair)
                      covered = True
          if covered : covered_pairs += 1
              
      return(coverage, covered_pairs)

  foundGoodPairs, numFoundGood = predicateCover(knownGoodPairs, predicates)
  foundBadPairs, numFoundBad = predicateCover(knownBadPairs, predicates)

  predicateSet = set(foundGoodPairs.keys()) | set(foundBadPairs.keys())

  filteredPredicateSet = set()
  for predicate in predicate :
    if len(foundBadPairs[predicate]) < eta :
      filteredPredicateSet.add(predicate)
      
  print "filteredPredicateSet: "
  print filteredPredicateSet
  
  expectedBadPairs = math.sqrt(len(train_pairs) / math.log(numGoodPairs))
  print "expectedBadPairs: ", expectedBadPairs
  
  filteredBadPairs = []
  for pair in knownBadPairs :
      if sum([pair in coveredpairs
              for coveredpairs
              in foundBadPairs.values()]) > expectedBadPairs :
        filteredBadPairs.append(pair)
  
  print "numGoodPairs: ", numGoodPairs
  print "numBadPairs: ", numBadPairs
      
  print "filteredBadPairs: "
  print len(filteredBadPairs)
  
  #print "knownBadPairs: "
  #print knownBadPairs
  
  filteredBadCoverage, numCoveredPairs = predicateCover(filteredBadPairs, predicates)


  
  print "filteredBadCoverage: "
  print filteredBadCoverage
  
  print "numCoveredPairs: ", numCoveredPairs
  
  epsilon = 1
  finalPredicateSet = []
  consideredPredicates = filteredPredicateSet
  while numGoodPairs >= epsilon :
    optimumCover = 0
    bestPredicate = None
    for predicate in consideredPredicates :
      cover = len(foundGoodPairs[predicate]) / float(len(filteredBadCoverage[predicate]))
      if cover > optimumCover :
        optimumCover = cover
        bestPredicate = predicate

    if not bestPredicate : break

    consideredPredicates.pop(bestPredicate)
    numGoodPairs -= len(foundGoodPairs[predicate])
    
    foundGoodPairs
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
  

if __name__ == '__main__':
  numTrainingPairs = 8000
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


