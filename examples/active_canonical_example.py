from dedupe import *
from canonical_example import init
from dedupe.training_sample import activeLearning, consoleLabel
from dedupe.blocking import trainBlocking, blockingIndex, mergeBlocks
from dedupe.predicates import *
from dedupe.core import scorePairs
from dedupe.clustering import cluster

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
training_data, training_pairs, data_model = activeLearning(data_d, data_model, consoleLabel, 20)

#training_pairs = randomTrainingPairs(data_d,
#                                     duplicates_s,
#                                     num_training_dupes,
#                                     num_training_distinct)

predicates = trainBlocking(training_pairs,
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


blocked_data = blockingIndex(data_d, predicates)
candidates = mergeBlocks(blocked_data)



print ""
print "Blocking reduced the number of comparisons by",
print int((1-len(candidates)/float(0.5*len(data_d)**2))*100),
print "%"
print "We'll make",
print len(candidates),
print "comparisons."

## training_data = trainingDistances(training_pairs, data_model)
## #print "training data from known duplicates: "
## #for instance in training_data :
## #  print instance

## print ""
## print "number of training items: "
## print len(training_data)
## print ""

## print "training weights ..."
## data_model = trainModel(training_data, numIterations, data_model)
## print ""

print "Learned Weights"
for k1, v1 in data_model.items() :
  try:
    for k2, v2 in v1.items() :
      print (k2, v2['weight'])
  except :
    print (k1, v1)

print ""

print "finding duplicates ..."
print ""

dupes = scoreDuplicates(candidates, data_d, data_model)
clustered_dupes = cluster(dupes, .2) 

# dupe_ids = set([frozenset(dupe_pair[0]) for dupe_pair in dupes])
# true_positives = dupe_ids & duplicates_s
# false_positives = dupe_ids - duplicates_s
# uncovered_dupes = duplicates_s - dupe_ids
# 
# print "False negatives" 
# for pair in uncovered_dupes :
#        print ""
#        for instance in tuple(pair) :
#          print data_d[instance].values()
# 
# print "____________________________________________"
# print "False positives" 
# 
# for pair in false_positives :
#   print ""
#   for instance in tuple(pair) :
#     print data_d[instance].values()
# 
# print ""
# 
# print "found duplicate"
# print len(dupes)
# 
# print "precision"
# print (len(dupes) - len(false_positives))/float(len(dupes))
# 
# print "recall"
# print  len(true_positives)/float(len(duplicates_s))
# print "ran in ", time.time() - t0, "seconds"
