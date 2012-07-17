from training_sample import activeLearning
from blocking import trainBlocking, blockingIndex, mergeBlocks
from predicates import *
import core
from random import sample
import clustering
import csv
import re
import os

def earlyChildhoodImport(filename) :
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
        
        data_d[i] = core.frozendict(instance)

    return(data_d, header)
    
    
def dataModel() :
  return  {'fields': 
            { 'Site name' : {'type': 'String', 'weight' : 0}, 
              'Address'   : {'type': 'String', 'weight' : 0},
              'Zip'       : {'type': 'String', 'weight' : 0}, 
              'Phone'     : {'type': 'String', 'weight' : 0}
            },
           'bias' : 0}

def init() :
  data_d, header = earlyChildhoodImport("examples/datasets/ECP_all_raw_input.csv")
  data_model = dataModel()
  return (data_d, data_model, header)

# user defined function to label pairs as duplicates or non-duplicates
def consoleLabel(uncertain_pairs, data_d) :
  duplicates = []
  nonduplicates = []

  for pair in uncertain_pairs :
    label = ''

    record_pair = [data_d[instance] for instance in pair]
    record_pair = tuple(record_pair)

    print "Site name: ", record_pair[0]['Site name'] 
    print "Address: ", record_pair[0]['Address'] 
    print "Zip: ", record_pair[0]['Zip'] 
    print "Phone: ", record_pair[0]['Phone'] 
    
    print ""
    
    print "Site name: ", record_pair[1]['Site name']
    print "Address: ", record_pair[1]['Address']
    print "Zip: ", record_pair[1]['Zip']
    print "Phone: ", record_pair[1]['Phone']
    
    #for instance in record_pair :
    #  print instance
  
    print ""
    print "Do these records refer to the same thing?"  

    valid_response = False
    while not valid_response :
      label = raw_input('(y)es / (n)o / (u)nsure\n')
      if label in ['y', 'n', 'u'] :
        valid_response = True

    if label == 'y' :
      duplicates.append(record_pair)
    elif label == 'n' :
      nonduplicates.append(record_pair)
    elif label != 'u' :
      print 'Nonvalid response'
      raise

  return({0:nonduplicates, 1:duplicates})

def dictSubset(d, keys) :
  return dict((k,d[k]) for k in keys if k in d)


num_training_dupes = 200
num_training_distinct = 16000
numIterations = 100
numTrainingPairs = 30

import time
t0 = time.time()
data_d, data_model, header = init()




print "importing data ..."

if os.path.exists('learned_settings.json') :
  data_model, predicates = core.readSettings('learned_settings.json')
else:
  #lets do some active learning here
  training_data, training_pairs, data_model = activeLearning(dictSubset(data_d, sample(data_d.keys(), 700)), data_model, consoleLabel, numTrainingPairs)

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
dupes = core.scoreDuplicates(candidates, data_d, data_model)
clustered_dupes = clustering.cluster(dupes, estimated_dupe_fraction = 0.4)

print "# duplicate sets"
print len(clustered_dupes)

orig_data = {}
with open("examples/datasets/ECP_all_raw_input.csv") as f :
  reader = csv.reader(f)
  reader.next()
  for row_id, row in enumerate(reader) :
    orig_data[row_id] = row
    

with open("examples/output/ECP_dupes_list_" + str(time.time()) + ".csv","w") as f :
  writer = csv.writer(f)
  heading_row = header
  heading_row.insert(0, "Group_ID")
  writer.writerow(heading_row)
  
  for group_id, cluster in enumerate(clustered_dupes, 1) :
    for candidate in sorted(cluster) :
      row = orig_data[candidate]
      row.insert(0, group_id)
      writer.writerow(row)


print "ran in ", time.time() - t0, "seconds"
