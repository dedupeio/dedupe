#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import csv
import exampleIO
import dedupe
import os
import time
import random
import optparse
import logging
from collections import defaultdict

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose == 1:
    log_level = logging.INFO
elif opts.verbose >= 2:
    log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)

# create a random set of training pairs based on known duplicates

def canonicalImport(filename):
    preProcess = exampleIO.preProcess
    data_d = {}
 
    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = [(k, preProcess(v)) for (k, v) in
                         row.iteritems()]
            data_d[filename + str(i)] = dedupe.core.frozendict(clean_row) 


    return data_d, reader.fieldnames


def evaluateDuplicates(found_dupes, true_dupes):
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)
    uncovered_dupes = true_dupes.difference(found_dupes)

    print 'found duplicate'
    print len(found_dupes)

    print 'precision'
    print 1 - len(false_positives) / float(len(found_dupes))

    print 'recall'
    print len(true_positives) / float(len(true_dupes))


settings_file = 'canonical_data_matching_learned_settings'

data_1, header = canonicalImport('tests/datasets/restaurant-1.csv')
data_2, _ = canonicalImport('tests/datasets/restaurant-2.csv')

training_pairs = dedupe.trainingDataLink(data_1, data_2, 'unique_id', 5000)
                                         
duplicates_s = set(frozenset(pair) for pair in training_pairs['match'])

t0 = time.time()

print 'number of known duplicate pairs', len(duplicates_s)

if os.path.exists(settings_file):
    deduper = dedupe.StaticRecordLink(settings_file)
else:
    fields = {'name': {'type': 'String'},
              'address': {'type': 'String'},
              'cuisine': {'type': 'String'},
              'city' : {'type' : 'String'}
              }

    deduper = dedupe.RecordLink(fields)
    deduper.sample(data_1, data_2, 100000) 
    deduper.markPairs(training_pairs)
    deduper.train()
    deduper.writeSettings(settings_file)


alpha = deduper.threshold(data_1, data_2)


# print candidates
print 'clustering...'
clustered_dupes = deduper.match(data_1, data_2, threshold=alpha)


print 'Evaluate Scoring'
found_dupes = set(frozenset((data_1[pair[0]], data_2[pair[1]])) 
                   for (pair, score) in deduper.matches
                   if score > alpha)

evaluateDuplicates(found_dupes, duplicates_s)

print 'Evaluate Clustering'

confirm_dupes = set(frozenset((data_1[pair[0]], data_2[pair[1]])) 
                    for pair in clustered_dupes)

evaluateDuplicates(confirm_dupes, duplicates_s)

print 'ran in ', time.time() - t0, 'seconds'
