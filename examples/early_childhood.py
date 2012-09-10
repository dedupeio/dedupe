#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import exampleIO
import dedupe
import time

input_file = 'examples/datasets/ECP_all_raw_input.csv'
output_file = 'examples/output/ECP_dupes_list.csv'
settings_file = 'ecp_learned_settings.json'
training_file = 'ecp_training.json'

t0 = time.time()

(data_d, header) = exampleIO.readData(input_file)

print 'importing data ...'

if os.path.exists(settings_file):
    deduper = dedupe.Dedupe(settings_file, 'settings file')
else:
    fields = {'Site name': {'type': 'String'},
              'Address': {'type': 'String'},
              'Zip': {'type': 'String'},
              'Phone': {'type': 'String'},
              }
    deduper = dedupe.Dedupe(fields, 'fields')

    if os.path.exists(training_file):

    # read in training json file

        deduper.readTraining(training_file)
        deduper.train()
    else:

    # get user input for active learning

        deduper.activeLearning(data_d,
                               dedupe.training_sample.consoleLabel)
        deduper.writeTraining(training_file)

deduper.findDuplicates(data_d)
deduper.writeSettings(settings_file)

clustered_dupes = deduper.duplicateClusters(threshold=.5)

print '# duplicate sets'
print len(clustered_dupes)
exampleIO.print_csv(input_file, output_file, header, clustered_dupes)

print 'ran in ', time.time() - t0, 'seconds'
