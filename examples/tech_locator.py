import csv
import re
import os

#dedupe modules
from dedupe.training_sample import activeLearning, consoleLabel
from dedupe.blocking import trainBlocking, blockingIndex, mergeBlocks
from dedupe.predicates import *
import dedupe.core
import dedupe.clustering

def techLocatorImport(filename) :
  data_d = {}
  duplicates_d = {}
  with open(filename) as f :
    reader = csv.reader(f, delimiter=',', quotechar='"')
    header = reader.next()
    for i, row in enumerate(reader) :
      instance = {}
      for j, col in enumerate(row) :
        col = re.sub('  +', ' ', col)
        col = re.sub('\n', ' ', col)
        instance[header[j]] = col.strip().strip('"').strip("'").lower()
        
        data_d[i] = dedupe.core.frozendict(instance)

    return(data_d, header)
    
    
def dataModel() :
  return  {'fields': 
            { 'OrganizationName' : {'type': 'String', 'weight' : 0}, 
              'Address'   : {'type': 'String', 'weight' : 0},
              'ZipCode'       : {'type': 'String', 'weight' : 0}, 
              'OrgPhone'     : {'type': 'String', 'weight' : 0}
            },
           'bias' : 0}

def init(inputFile) :
  data_d, header = techLocatorImport(inputFile)
  data_model = dataModel()
  return (data_d, data_model, header)

# user defined function to label pairs as duplicates or non-duplicates


def dictSubset(d, keys) :
  return dict((k,d[k]) for k in keys if k in d)


inputFile = "examples/datasets/Tech Locator Master List.csv"
num_training_dupes = 200
num_training_distinct = 16000
numIterations = 100
numTrainingPairs = 30

import time
t0 = time.time()
data_d, data_model, header = init(inputFile)


print "importing data ..."

if os.path.exists('learned_settings.json') :
  data_model, predicates = core.readSettings('learned_settings.json')
else:
  #lets do some active learning here
  training_data, training_pairs, data_model = activeLearning(data_d, data_model, consoleLabel, numTrainingPairs)

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

  core.writeSettings('learned_settings.json',
                     data_model,
                     predicates)


blocked_data = blockingIndex(data_d, predicates)
candidates = mergeBlocks(blocked_data)

print ""
print "Blocking reduced the number of comparisons by",
print int((1-len(candidates)/float(0.5*len(data_d)**2))*100),
print "%"
print "We'll make",
print len(candidates),
print "comparisons."

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
dupes = core.scoreDuplicates(candidates, data_d, data_model, .5)
clustered_dupes = clustering.cluster(dupes, estimated_dupe_fraction = 0.4)

print "# duplicate sets"
print len(clustered_dupes)

orig_data = {}
with open(inputFile) as f :
  reader = csv.reader(f)
  reader.next()
  for row_id, row in enumerate(reader) :
    orig_data[row_id] = row
    

with open("examples/output/TL_dupes_list_" + str(time.time()) + ".csv","w") as f :
  writer = csv.writer(f)
  heading_row = header
  heading_row.insert(0, "Group_ID")
  writer.writerow(heading_row)
  
  dupe_id_list = []
  
  for group_id, cluster in enumerate(clustered_dupes, 1) :
    for candidate in sorted(cluster) :
      dupe_id_list.append(candidate)
      row = orig_data[candidate]
      row.insert(0, group_id)
      writer.writerow(row)
      
  for id in orig_data :
    if not id in set(dupe_id_list) :
      row = orig_data[id]
      row.insert(0, 'x')
      writer.writerow(row)


print "ran in ", time.time() - t0, "seconds"
