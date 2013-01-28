#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import exampleIO
import dedupe
import time

#input_file = 'examples/datasets/ECP_all_raw_input.csv'
input_file = 'examples/datasets/ECP_all_normalized_input.csv'
output_file = 'examples/output/ECP_dupes_list.csv'
settings_file = 'csv_example_learned_settings.json'
training_file = 'csv_example_training.json'

t0 = time.time()

print 'importing data ...'
(data_d, header) = exampleIO.readData(input_file)

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
        deduper.train(data_d, training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # get user input for active learning
    deduper.train(data_d, dedupe.training_sample.consoleLabel)
    deduper.writeTraining(training_file)


print 'blocking...'
blocker = deduper.blockingFunction()
blocked_data = dedupe.blocking.blockingIndex(data_d, blocker)
print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, cluster_threshold=.5)
deduper.writeSettings(settings_file)

print '# duplicate sets'
print len(clustered_dupes)
exampleIO.print_csv(input_file, output_file, header, clustered_dupes)

print 'ran in ', time.time() - t0, 'seconds'