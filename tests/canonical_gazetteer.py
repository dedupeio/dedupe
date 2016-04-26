#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from future.utils import viewitems

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
if opts.verbose :
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
                         viewitems(row)]
            data_d[filename + str(i)] = dedupe.core.frozendict(clean_row) 


    return data_d, reader.fieldnames


def evaluateDuplicates(found_dupes, true_dupes):
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)
    uncovered_dupes = true_dupes.difference(found_dupes)

    print('found duplicate')
    print(len(found_dupes))

    print('precision')
    print(1 - len(false_positives) / float(len(found_dupes)))

    print('recall')
    print(len(true_positives) / float(len(true_dupes)))


settings_file = 'canonical_data_matching_learned_settings'

data_1, header = canonicalImport('tests/datasets/restaurant-1.csv')
data_2, _ = canonicalImport('tests/datasets/restaurant-2.csv')

training_pairs = dedupe.trainingDataLink(data_1, data_2, 'unique_id', 5000)
                                         
duplicates_s = set(frozenset(pair) for pair in training_pairs['match'])

t0 = time.time()

print('number of known duplicate pairs', len(duplicates_s))

if os.path.exists(settings_file):
    with open(settings_file, 'rb') as f :
        gazetteer = dedupe.StaticGazetteer(f)
else:
    fields = [{'field': 'name', 'type': 'String'},
              {'field': 'address', 'type': 'String'},
              {'field': 'cuisine', 'type': 'String'},
              {'field': 'city','type' : 'String'}
              ]

    gazetteer = dedupe.Gazeteer(fields)
    gazetteer.sample(data_1, data_2, 10000) 
    gazetteer.markPairs(training_pairs)
    gazetteer.train()
    
if not gazetteer.blocked_records:
    gazetteer.index(data_2)

with open(settings_file, 'wb') as f:
    gazetteer.writeSettings(f, index=True)
        
alpha = gazetteer.threshold(data_1)


# print candidates
print('clustering...')
clustered_dupes = gazetteer.match(data_1, threshold=alpha, n_matches=1)

print('Evaluate Clustering')
confirm_dupes = set(frozenset((data_1[pair[0]], data_2[pair[1]]))
                    for matches in clustered_dupes
                    for pair, score in matches)

evaluateDuplicates(confirm_dupes, duplicates_s)

print('ran in ', time.time() - t0, 'seconds')

