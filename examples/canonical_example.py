from itertools import combinations
import csv
import re

#dedupe modules
from dedupe import *
from dedupe.core import frozendict
from dedupe.clustering.chaudhi import cluster

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
                # we may want to think about removing common stop
                # words
                #col = col.strip()
                #col = re.sub('[^a-z0-9 ]', ' ', col)
                #col = re.sub('\.', ' ', col)
                #col = re.sub(r'\bthe\b', ' ', col)
                #col = re.sub(r'restaurant', ' ', col)
                #col = re.sub(r'cafe', ' ', col)
                #col = re.sub(r'diner', ' ', col)
                #col = re.sub(r'\(.*\)', ' ', col)
                
                #col = re.sub(r'\bn\.', ' ', col)
                #col = re.sub(r'\bs\.', ' ', col)
                #col = re.sub(r'\be\.', ' ', col)
                #col = re.sub(r'\bw\.', ' ', col)
                col = re.sub(r'\broad\b', 'rd', col)
                col = re.sub('  +', ' ', col)
                

                instance[header[j]] = col.strip().strip('"').strip("'")
                
            data_d[i] = frozendict(instance)

    duplicates_s = set([])
    for unique_id in duplicates_d :
      if len(duplicates_d[unique_id]) > 1 :
        for pair in combinations(duplicates_d[unique_id], 2) :
          duplicates_s.add(frozenset(pair))

    return(data_d, header, duplicates_s)

def dataModel() :
  return  {'fields': 
            {'name' : {'type': 'String', 'weight' : 0}, 
             'address' : {'type' :'String', 'weight' : 0},
             'city' : {'type': 'String', 'weight' : 0},
             'cuisine' : {'type': 'String', 'weight' : 0},
#             'name:city' : {'type': 'Interaction', 'interaction-terms': ['name', 'city'], 'weight' : 0}
            },
           'bias' : 0}


def init() :
    
  data_d, header, duplicates_s = canonicalImport("examples/datasets/restaurant-nophone-training.csv")
  data_model = dataModel()
  return (data_d, duplicates_s, data_model)
  
# main execution
if __name__ == '__main__':
    
  from dedupe import *
  from dedupe.predicates import *
  import os

  import time
  t0 = time.time()
  num_training_dupes = 200
  num_training_distinct = 1600
  numIterations = 30

  (data_d, duplicates_s, data_model) = init()

  print "number of duplicates pairs"
  print len(duplicates_s)
  print ""

  if os.path.exists('restaurant_learned_settings.json') :
    data_model, predicates = core.readSettings('restaurant_learned_settings.json')

  else :
      
    training_pairs = training_sample.randomTrainingPairs(data_d,
                                                         duplicates_s,
                                                         num_training_dupes,
                                                         num_training_distinct)

    

    predicates = blocking.trainBlocking(training_pairs,
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
  
    training_data = training_sample.addTrainingData(training_pairs, data_model)


    print ""
    print "number of training items: "
    print len(training_data)
    print ""

    ## alpha = crossvalidation.gridSearch(training_data,
    ##                                    core.trainModel,
    ##                                    data_model)

    alpha = .01

    print "training weights ..."
    data_model = core.trainModel(training_data, numIterations, data_model, alpha)
    print ""

    #core.writeSettings('restaurant_learned_settings.json', data_model, predicates)

  print "Learned Weights"
  for k1, v1 in data_model.items() :
    try:
      for k2, v2 in v1.items() :
        print (k2, v2['weight'])
    except :
      print (k1, v1)

  blocked_data = blocking.blockingIndex(data_d, predicates)
  candidates = blocking.mergeBlocks(blocked_data)

  print ""
  print "Blocking reduced the number of comparisons by",
  print int((1-len(candidates)/float(0.5*len(data_d)**2))*100),
  print "%"
  print "We'll make",
  print len(candidates),
  print "comparisons."


  print ""
  
  dupes = core.scoreDuplicates(candidates, data_d, data_model, .50)

  #print dupes

  dupe_ids = set([frozenset(dupe_pair[0]) for dupe_pair in dupes])
  true_positives = dupe_ids & duplicates_s
  false_positives = dupe_ids - duplicates_s
  uncovered_dupes = duplicates_s - dupe_ids

 #  print "False negatives" 
#   for pair in uncovered_dupes :
#       print ""
#       for instance in tuple(pair) :
#           print data_d[instance].values()
#           
#   print "____________________________________________"
#   print "False positives" 
# 
#   for pair in false_positives :
#       print ""
#       for instance in tuple(pair) :
#           print data_d[instance].values()
# 
#   print ""

  #print "found duplicate"
  #print len(dupes)
  
  print "precision"
  print (len(dupes) - len(false_positives))/float(len(dupes))

  print "recall"
  print  len(true_positives)/float(len(duplicates_s))
  print "ran in ", time.time() - t0, "seconds"
  
  # print "finding duplicates ..."
#   print ""
#   dupes = core.scoreDuplicates(candidates, data_d, data_model)
#   clustered_dupes = cluster(dupes,
#                             estimated_dupe_fraction = .2)
# 
#   
#   confirm_dupes = set([])
#   for dupe_set in clustered_dupes :
#     if (len(dupe_set) == 2) :
#       confirm_dupes.add(frozenset(dupe_set))
#     else :
#       for pair in combinations(dupe_set, 2) :
#         confirm_dupes.add(frozenset(pair))
# 
#   #dupe_ids = set([frozenset(dupe_pair[0]) for dupe_pair in dupes])
#   true_positives = confirm_dupes & duplicates_s
#   false_positives = confirm_dupes - duplicates_s
#   uncovered_dupes = duplicates_s - confirm_dupes
# 
#   print "False negatives" 
#   for pair in uncovered_dupes :
#       print ""
#       for instance in tuple(pair) :
#           print data_d[instance].values()
#           
#   print "____________________________________________"
#   print "False positives" 
# 
#   for pair in false_positives :
#       print ""
#       for instance in tuple(pair) :
#           print data_d[instance].values()
# 
#   print ""
# 
#   print "found duplicate"
#   print len(confirm_dupes)
#   
#   print "precision"
#   print (len(confirm_dupes) - len(false_positives))/float(len(confirm_dupes))
# 
#   print "recall"
#   print  len(true_positives)/float(len(duplicates_s))
#   print "ran in ", time.time() - t0, "seconds"
