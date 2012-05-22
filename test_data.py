from itertools import combinations
import csv
import re

def canonicalImport(filename) :


    data_d = {}
    duplicates_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        for i, row in enumerate(reader) :
            instance = {}
            for j, col in enumerate(row) :
              if header[j] == 'unique_id' :
                duplicates_d.setdefault(col, []).append(i)
              else :
                #col = col.strip()
                #col = re.sub('[^a-z0-9 ]', ' ', col)
                #col = re.sub('  +', ' ', col)
                instance[header[j]] = col.strip().strip('"').strip("'")
                
            data_d[i] = instance

    duplicates_s = set([])
    for unique_id in duplicates_d :
      if len(duplicates_d[unique_id]) > 1 :
        for pair in combinations(duplicates_d[unique_id], 2) :
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


def init() :
  data_d, header, duplicates_s = canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  return (data_d, duplicates_s, data_model)

if __name__ == '__main__':
  (data_d, duplicates_s, data_model) = init()
  
  
