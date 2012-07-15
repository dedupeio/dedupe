from training_sample import activeLearning
from blocking import trainBlocking, blockingIndex, mergeBlocks
from predicates import *
from core import scoreDuplicates
from random import sample
from clustering import cluster 
from itertools import combinations
import csv
import re
from core import frozendict

def earlyChildhoodImport(filename) :

    original_data_d = {}
    data_d = {}
    duplicates_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        for i, row in enumerate(reader) :
            instance_orig = {}
            instance = {}
            for j, col in enumerate(row) :
              col = re.sub('  +', ' ', col)
              col = re.sub('\n', ' ', col)
              instance_orig[header[j]] = col.strip().strip('"').strip("'")
              instance[header[j]] = col.strip().strip('"').strip("'").lower()
            
            original_data_d[i] = frozendict(instance_orig)    
            data_d[i] = frozendict(instance)
            #print data_d[i]

    return(original_data_d, data_d, header)
    
    
             # 'Id' : {'type': 'String', 'weight' : 0}, 
#              'Site Name' : {'type': 'String', 'weight' : 0}, 
#              'Program Name' : {'type': 'String', 'weight' : 0}, 
#              'Length of Day' : {'type' :'String', 'weight' : 0},
#              'Address' : {'type': 'String', 'weight' : 0},
#              'Phone' : {'type': 'String', 'weight' : 0}

def dataModel() :
  return  {'fields': 
            { 'Site name' : {'type': 'String', 'weight' : 0}, 
              'Address' : {'type': 'String', 'weight' : 0},
              'Zip' : {'type': 'String', 'weight' : 0}, 
              'Phone' : {'type': 'String', 'weight' : 0}
            },
           'bias' : 0}


def init() :
    
  #data_d, header = earlyChildhoodImport("source/csv/CPS_Early_Childhood_Portal_scrape.csv")
  original_data_d, data_d, header = earlyChildhoodImport("datasets/ECP_all_raw_input.csv")
  
  #print data_d
  
  data_model = dataModel()
  return (original_data_d, data_d, data_model)


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

def printToCsv(clustered_dupes, original_data_d) :
  print "writing to csv"
  FILE = open("output/ECP_dupes_list_" + str(time.time()) + ".csv","w")
  output = "\"Group id\",\"Id\",\"Source\",\"Site name\",\"Address\",\"Zip\",\"Phone\",\"Fax\",\"Program Name\",\"Length of Day\",\"IDHS Provider ID\",\"Agency\",\"Neighborhood\",\"Funded Enrollment\",\"Program Option\",\"Number per Site EHS\",\"Number per Site HS\",\"Director\",\"Head Start Fund\",\"Early Head Start Fund\",\"CC fund\",\"Progmod\",\"Website\",\"Executive Director\",\"Center Director\",\"ECE Available Programs\",\"NAEYC Valid Until\",\"NAEYC Program Id\",\"Email Address\",\"Ounce of Prevention Description\",\"Purple binder service type\"\n"
  FILE.write(output)
  
  #print out all found dupes
  dupe_id_list = []
  i = 1
  row_cnt = 0
  for dupe_set in clustered_dupes :
    for dupe_id in dupe_set :
      item = original_data_d[dupe_id]
      dupe_id_list.append(dupe_id)
      FILE.write(printRow(item,i))
      row_cnt += 1
    i += 1
      
  #print the rest that weren't found
  dupe_id_list = set(dupe_id_list)
  #print "dupe ids"
  #print dupe_id_list
  for row in original_data_d :
    #print row
    #print "row in dupes?", (not row in dupe_id_list)
    if not row in dupe_id_list :
      #print "adding"
      FILE.write(printRow(original_data_d[row],i))
      i += 1
      row_cnt += 1
  
  FILE.close()
  print len(original_data_d), "input rows"
  print len(clustered_dupes), "dupe clusters found"
  print i, "groups printed"
  print row_cnt, "rows printed"


def printRow(item, i) :
  output = str(i) + ","
  output += "\"" + item['Id'] + "\","
  output += "\"" + item['Source'] + "\","
  output += "\"" + item['Site name'] + "\","
  output += "\"" + item['Address'] + "\","
  output += "\"" + item['Zip'] + "\","
  output += "\"" + item['Phone'] + "\","
  output += "\"" + item['Fax'] + "\","
  output += "\"" + item['Program Name'] + "\","
  output += "\"" + item['Length of Day'] + "\","
  output += "\"" + item['IDHS Provider ID'] + "\","
  output += "\"" + item['Agency'] + "\","
  output += "\"" + item['Neighborhood'] + "\","
  output += "\"" + item['Funded Enrollment'] + "\","
  output += "\"" + item['Program Option'] + "\","
  output += "\"" + item['Number per Site EHS'] + "\","
  output += "\"" + item['Number per Site HS'] + "\","
  output += "\"" + item['Director'] + "\","
  output += "\"" + item['Head Start Fund'] + "\","
  output += "\"" + item['Eearly Head Start Fund'] + "\","
  output += "\"" + item['CC fund'] + "\","
  output += "\"" + item['Progmod'] + "\","
  output += "\"" + item['Website'] + "\","
  output += "\"" + item['Executive Director'] + "\","
  output += "\"" + item['Center Director'] + "\","
  output += "\"" + item['ECE Available Programs'] + "\","
  output += "\"" + item['NAEYC Valid Until'] + "\","
  output += "\"" + item['NAEYC Program Id'] + "\","
  output += "\"" + item['Email Address'] + "\","
  output += "\"" + item['Ounce of Prevention Description'] + "\","
  output += "\"" + item['Purple binder service type'] + "\","
  output += "\n"
  
  return output

num_training_dupes = 200
num_training_distinct = 16000
numIterations = 100
numTrainingPairs = 30
percentEstimatedDupes = .7
numNearestNeighbors = 8

import time
t0 = time.time()
(original_data_d, data_d, data_model) = init()

print "importing data ..."
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
dupes = scoreDuplicates(candidates, data_d, data_model)
clustered_dupes = cluster(dupes, percentEstimatedDupes, numNearestNeighbors)

print "# duplicates"
print len(clustered_dupes)

printToCsv(clustered_dupes, original_data_d)

print "ran in ", time.time() - t0, "seconds"