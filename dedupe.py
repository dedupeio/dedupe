def canonicalImport(filename) :
    import csv

    data_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        print header
        for i, row in enumerate(reader) :
            instance = {}
            for j, col in enumerate(row) :
                instance[header[j]] = col.strip()
            data_d[i] = instance

    return(data_d, header)

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
  import itertools
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
  data_d, header = canonicalImport("./datasets/restaurant-nophone.csv")
  data_model = dataModel()
  candidates = identifyCandidates(data_d)
  print "finding duplicates"
  dupes = findDuplicates(candidates, data_d, data_model, 5)
  #print dupes
