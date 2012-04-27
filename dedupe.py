import itertools

def canonicalImport(filename) :
    import csv

    data_d = {}
    training_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        print header
        for i, row in enumerate(reader) :
            instance = {}
            for j, col in enumerate(row) :
              if header[j] == 'unique_id' :
                training_d.setdefault(col, []).append(i)
              else :
                instance[header[j]] = col.strip()
                
            data_d[i] = instance

    training_l = []
    for unique_id in training_d :
      if len(training_d[unique_id]) > 1 :
        for pair in itertools.combinations(training_d[unique_id], 2) :
          training_l.append(pair)

    return(data_d, header, training_l)

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
  import distance #libdistance library http://monkey.org/~jose/software/libdistance/
  duplicateScores = []

  for candidates_set in candidates :
    for pair in itertools.combinations(candidates_set, 2):
      scorePair = {}
      score = data_model['bias'] 
      fields = data_model['fields']
      for name in fields :
        if fields[name]['type'] == 'String' :
          distanceFunc = distance.levenshtein
        x = distanceFunc(data_d[pair[0]][name],data_d[pair[1]][name])
        score += x * fields[name]['weight']
      scorePair[pair] = score
      #print (pair, score)
      if score < threshold :
        print (data_d[pair[0]],data_d[pair[1]])
        print score
        duplicateScores.append(scorePair)
  
  return duplicateScores

if __name__ == '__main__':
  data_d, header, training_l = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  #print "training data: "
  #print training_l
  print "number of duplicates from training data "
  print len(training_l)
  print "finding duplicates ..."
  dupes = findDuplicates(candidates, data_d, data_model, 5)
  print dupes
