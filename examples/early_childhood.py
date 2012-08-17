import csv
import re
import os
import json
import dedupe.dedupe
import dedupe.training_sample

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
        
        data_d[i] = dedupe.core.frozendict(instance)

    return(data_d, header)
    
def init(inputFile) :
  data_d, header = earlyChildhoodImport(inputFile)
  return (data_d, header)

#create our deduper object  
deduper = dedupe.dedupe.Dedupe() #<- irony
 
deduper.data_model = {'fields': 
            { 'Site name' : {'type': 'String', 'weight' : 0}, 
              'Address'   : {'type': 'String', 'weight' : 0},
              'Zip'       : {'type': 'String', 'weight' : 0},
              'Phone'     : {'type': 'String', 'weight' : 0},
#              'SiteName:Address' : {'type': 'Interaction', 'interaction-terms': ['Site name', 'Address'], 'weight' : 0}
            },
           'bias' : 0}

inputFile = "examples/datasets/ECP_all_raw_input.csv"
learnedSettingsFile = "ecp_learned_settings.json"
trainingFile = "ecp_training.json"

import time
t0 = time.time()
data_d, header = init(inputFile)


print "importing data ..."

if os.path.exists(learnedSettingsFile) :
  deduper.readSettings(learnedSettingsFile)
else:
  if os.path.exists(trainingFile) :
    #read in training json file
    deduper.readTraining(trainingFile)
    deduper.findAlpha()
    deduper.train()
  else :  
    #get user input for active learning
    deduper.activeLearning(data_d, dedupe.training_sample.consoleLabel) 
    deduper.writeTraining(trainingFile)

deduper.mapBlocking(data_d)  
deduper.identifyCandidates()

print ""
print "Blocking reduced the number of comparisons by",
print int((1-len(deduper.candidates)/float(0.5*len(data_d)**2))*100),
print "%"
print "We'll make",
print len(deduper.candidates),
print "comparisons."

print "Learned Weights"
for k1, v1 in deduper.data_model.items() :
  try:
    for k2, v2 in v1.items() :
      print (k2, v2['weight'])
  except :
    print (k1, v1)

print ""

print "finding duplicates ..."
print ""
deduper.score(data_d)

## for pair, score in deduper.dupes :
##   if 6 in (pair[0], pair[1]) :
##     print "Score :", score
##     print pair
##     for k in ['Site name', 'Address'] :
##       print data_d[pair[0]][k]
##       print data_d[pair[1]][k]
##     print 

#clustered_dupes = dedupe.clustering.chaudhi.cluster(deduper.dupes, estimated_dupe_fraction = .9)
clustered_dupes = deduper.duplicateClusters(.5)

print "# duplicate sets"
print len(clustered_dupes)

orig_data = {}
with open(inputFile) as f :
  reader = csv.reader(f)
  reader.next()
  for row_id, row in enumerate(reader) :
    orig_data[row_id] = row
    

#with open("examples/output/ECP_dupes_list_" + str(time.time()) + ".csv","w") as f :
with open("examples/output/ECP_dupes_list.csv","w") as f :
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
    if id not in set(dupe_id_list) :
      row = orig_data[id]
      row.insert(0, 'x')
      writer.writerow(row)


print "ran in ", time.time() - t0, "seconds"
