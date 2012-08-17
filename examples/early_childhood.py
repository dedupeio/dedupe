import csv
import re
import os
import print_csv
import dedupe.dedupe
import dedupe.core
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


input_file = "examples/datasets/ECP_all_raw_input.csv"
output_file = "examples/output/ECP_dupes_list.csv"
learned_settings_file = "ecp_learned_settings.json"
training_file = "ecp_training.json"

import time
t0 = time.time()
data_d, header = earlyChildhoodImport(input_file)

print "importing data ..."

#create our deduper object  
deduper = dedupe.dedupe.Dedupe() #<- irony
 
if os.path.exists(learned_settings_file) :
  deduper.readSettings(learned_settings_file)
else:
  deduper.data_model = {'fields': 
    { 'Site name' : {'type': 'String', 'weight' : 0}, 
      'Address'   : {'type': 'String', 'weight' : 0},
      'Zip'       : {'type': 'String', 'weight' : 0},
      'Phone'     : {'type': 'String', 'weight' : 0},
      #'SiteName:Address' : {'type': 'Interaction', 'interaction-terms': ['Site name', 'Address'], 'weight' : 0}
    },
   'bias' : 0}

  if os.path.exists(training_file) :
    #read in training json file
    deduper.readTraining(training_file)
    deduper.findAlpha()
    deduper.train()
  else :  
    #get user input for active learning
    deduper.activeLearning(data_d, dedupe.training_sample.consoleLabel) 
    deduper.writeTraining(training_file)

deduper.printLearnedWeights()
deduper.findDuplicates(data_d)

#clustered_dupes = dedupe.clustering.chaudhi.cluster(deduper.dupes, estimated_dupe_fraction = .9)
clustered_dupes = deduper.duplicateClusters(.5)

print "# duplicate sets"
print len(clustered_dupes)
print_csv.print_csv(input_file, output_file, header, clustered_dupes)
print "ran in ", time.time() - t0, "seconds"