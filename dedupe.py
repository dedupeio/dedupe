import itertools
import distance #libdistance library http://monkey.org/~jose/software/libdistance/

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

    duplicates_l = []
    for unique_id in duplicates_d :
      if len(duplicates_d[unique_id]) > 1 :
        for pair in itertools.combinations(duplicates_d[unique_id], 2) :
          duplicates_l.append(pair)

    return(data_d, header, duplicates_l)

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
      if score < threshold :
        print (data_d[pair[0]],data_d[pair[1]])
        print score
        duplicateScores.append(scorePair)
  
  return duplicateScores

def calculateDistance(instance_1, instance_2, fields) :
  distances_d = {}
  for name in fields :
    if fields[name]['type'] == 'String' :
      distanceFunc = distance.levenshtein
    x = distanceFunc(instance_1[name],instance_2[name])
    distances_d[name] = x

  return distances_d

def createTrainingPairs(data_d, duplicates_l, n) :
  import random
  train_pairs = {}
  for pair in duplicates_l :
    train_pairs[pair] = 1
  while len(train_pairs) < n :
    random_pair = tuple(random.sample(data_d.keys(), 2))
    if random_pair not in train_pairs : 
      train_pairs[random_pair] = 0
      
  return(train_pairs)

def createTrainingData(data_d, duplicates_l, n, data_model) :
  train_pairs = createTrainingPairs(data_d, duplicates_l, n)

  training_data = []
  for pair in train_pairs :
    distances = calculateDistance(data_d[pair[0]], data_d[pair[1]], data_model['fields'])
    training_data.append((distances, train_pairs[pair]))

  return training_data

if __name__ == '__main__':
  data_d, header, duplicates_l = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print duplicates_l
  
  print "number of known duplicates: "
  print len(duplicates_l)

  train_pairs = createTrainingData(data_d, duplicates_l, 500, data_model)
  print "training data from known duplicates: "
  print train_pairs
  print "number of training items: "
  print len(train_pairs)

  print "finding duplicates ..."
  dupes = findDuplicates(candidates, data_d, data_model, 5)
  #print dupes
