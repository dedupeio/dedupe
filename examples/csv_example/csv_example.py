#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import csv
import re
import dedupe
from AsciiDammit import asciiDammit
from collections import defaultdict

def preProcess(column) :
  column = asciiDammit(column)
  column = re.sub('  +', ' ', column)
  column = re.sub('\n', ' ', column)
  column = column.strip().strip('"').strip("'").lower().strip()
  return column

def readData(filename) :
  data_d = {}
  with open(filename) as f :
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for i, row in enumerate(reader) :
      clean_row = [(k, preProcess(v)) for k,v in row.iteritems()]
      data_d[i] = dedupe.core.frozendict(clean_row)
      
  return(data_d, reader.fieldnames)

os.chdir('./examples/csv_example/')

input_file = 'csv_example_messy_input.csv'
output_file = 'csv_example_output.csv'
settings_file = 'csv_example_learned_settings.json'
training_file = 'csv_example_training.json'

print 'importing data ...'
(data_d, header) = readData(input_file)

# take a sample of data_d for training
data_d_sample = dedupe.core.sampleDict(data_d, 700)

if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
    fields = {'Site name': {'type': 'String'},
              'Address': {'type': 'String'},
              'Zip': {'type': 'String'},
              'Phone': {'type': 'String'},
              }
    deduper = dedupe.Dedupe(fields)

    if os.path.exists(training_file):
        # read in training json file
        print 'reading labeled examples from ', training_file
        deduper.train(data_d_sample, training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # get user input for active learning
    deduper.train(data_d_sample, dedupe.training_sample.consoleLabel)
    deduper.writeTraining(training_file)

print 'blocking...'
blocker = deduper.blockingFunction()
blocked_data = dedupe.blocking.blockingIndex(data_d, blocker)
print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, cluster_threshold=.5)
deduper.writeSettings(settings_file)

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