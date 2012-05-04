import itertools
#import distance #libdistance library http://monkey.org/~jose/software/libdistance/
import affinegap
import lr

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
                instance[header[j]] = col.strip()
                
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
      distanceFunc = affinegap.affineGapDistance
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

  return training_data

def trainModel(training_data, iterations, data_model) :
    trainer = lr.LogisticRegression()
    trainer.train(training_data, iterations)

    data_model['bias'] = trainer.bias
    for name in data_model['fields'] :
        data_model['fields'][name]['weight'] = trainer.weight[name]

    return(data_model)

if __name__ == '__main__':
  data_d, header, duplicates_s = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print duplicates_s
  
  print "number of known duplicates: "
  print len(duplicates_s)

  training_data = createTrainingData(data_d, duplicates_s, 500, data_model)
  #print "training data from known duplicates: "
  #print training_data
  print "number of training items: "
  print len(training_data)

  data_model = trainModel(training_data, 100, data_model)
  
  print "finding duplicates ..."
  dupes = findDuplicates(candidates, data_d, data_model, -.5)
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
