#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The following code demonstrates how to use dedupe with a flat (CSV) file. All operations are performed in memory, so it won't work for data sets that are larger than ~10,000 rows. 
"""

import os
import csv
import re
from collections import defaultdict
from AsciiDammit import asciiDammit
import dedupe

# We start with a csv file with our messy data. In this example, it is listings of early childhood education centers in Chicago compiled from several different sources. The output file will also be a csv, but with our clustered results.
os.chdir('./examples/csv_example/')
input_file = 'csv_example_messy_input.csv'
output_file = 'csv_example_output.csv'
settings_file = 'csv_example_learned_settings.json'
training_file = 'csv_example_training.json'

def preProcess(column) :
  """
  Our goal here is to find meaningful duplicates, so things like casing, extra spaces, quotes and new lines can be ignored. preProcess removes these.
  """
  column = asciiDammit(column)
  column = re.sub('  +', ' ', column)
  column = re.sub('\n', ' ', column)
  column = column.strip().strip('"').strip("'").lower().strip()
  return column

def readData(filename) :
  """
  We read in the data from a CSV file and as we do, we preProcess it. The output is a dictionary of records, where the key is a unique record ID (created with enumerate()) and each value is a frozendict (basically a hashable dictionary) of the row fields.
  """
  data_d = {}
  with open(filename) as f :
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for i, row in enumerate(reader) :
      clean_row = [(k, preProcess(v)) for k,v in row.iteritems()]
      data_d[i] = dedupe.core.frozendict(clean_row)
      
  return(data_d, reader.fieldnames)

print 'importing data ...'
(data_d, header) = readData(input_file)

# In order to train dedupe, we need to compare some records. We can't compare them all, because the number of possible combinations can be much too large (~0.5*N^2). We take a random sample of all possible pairs.
data_sample = dedupe.core.dataSample(data_d, 150000)


# If the settings files, which we mentioned above, exists, then we read it in. Passing in a settings file is one of the three ways to initialize a dedupe instance.
if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)

# Alternately, we can initialize a dedupe instance by declaring a field definition for the data. This defines the fields that we want to compare and how to compare them.

# A field definition is a dictionary where the keys are the fields that will be used for training a model and the values are the field specification.

# Field types include
# - String

#A 'String' type field must have as its key a name of a field as it appears in the data dictionary and a type declaration ex. {'Phone': {type: 'String'}}
else:
  fields = {'Site name': {'type': 'String'},
            'Address': {'type': 'String'},
            'Zip': {'type': 'String'},
            'Phone': {'type': 'String'},
            }
  deduper = dedupe.Dedupe(fields)

  # Dedupe will learn to predict if two records are duplicates based upon their similarity. In this example, that similarity is a weighted combination of the field by field [string similarity](http://en.wikipedia.org/wiki/String_metric) between records. Dedupe learns these weights.

  # Dedupe will ask a user to label pairs of records as duplicates or not. These labeled records are saved and can be reused for later training. To train dedupe with these examples, call deduper.train as shown.
  if os.path.exists(training_file) :
    print 'reading labeled examples from ', training_file
    deduper.train(data_sample, training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'

  # Dedupe can actively learn, that means it will select the records it is most uncertain about and will ask the user to label it. It will then learn from that labeling, update, and ask for the next most uncertain pair.

  # To do this, train method requires that you pass it a function to do this labeling, in this case, consoleLabel.

  # For consoleLabel, use 'y', 'n' and 'u' keys to flag duplicates, 'f' when you are finished.
  else:
      deduper.train(data_sample, dedupe.training_sample.consoleLabel)

      # Save away our labeled training pairs to a JSON file.
      deduper.writeTraining(training_file)

# Now that dedupe knows how to compare records, we use the same training data to block records in to groups. The goal is to reduce the total number of comparisons.

# blockingFunction learns the blocking rules, if not already defined in settings_file and returns a function. That function will take a record and return all the blocks it will fit in to.
print 'blocking...'
blocker = deduper.blockingFunction()
# Save our settings file, which includes learned weights and blcoking rules.
deduper.writeSettings(settings_file)
# blockingIndex loads all the original data in to memory and places them in to blocks. Each record can be blocked in many ways, so for larger data, memory will be a limiting factor.
blocked_data = dedupe.blocking.blockingIndex(data_d, blocker)

# duplicateClusters will return sets of record IDs that dedupe believes are all referrin to the same entity.

# pairwise_threshold -- Number between 0 and 1 (default is .5). We will only 
#                       consider as duplicates  ecord pairs as duplicates if 
#                       their estimated duplicate likelihood is greater than 
#                       the pairwise threshold. 
# cluster_threshold --  Number between 0 and 1 (default is .5). Lowering the 
#                       number will increase precision, raising it will increase
#                       recall
print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, cluster_threshold=.5)

# Now that we have our clustered duplicates, we write our original data back out to a CSV with a new column called 'Cluster ID' which indicates which records refer to each other.
print '# duplicate sets', len(clustered_dupes)
orig_data = {}

cluster_membership = defaultdict(lambda:'x')
for cluster_id, cluster in enumerate(clustered_dupes) :
  for record_id in cluster :
    cluster_membership[record_id] = cluster_id

f_input = open(input_file) 
reader = csv.reader(f_input)
reader.next()
    
with open(output_file,"w") as f :
  writer = csv.writer(f)
  heading_row = header
  heading_row.insert(0, "Cluster ID")
  writer.writerow(heading_row)

  for i, row in enumerate(reader) :
    cluster_id = cluster_membership[i]
    row.insert(0, cluster_id)
    writer.writerow(row)

f_input.close()
