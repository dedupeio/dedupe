#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from future.utils import viewitems
from builtins import range

from itertools import combinations
import csv
import exampleIO

import dedupe
import os
import time
import optparse
import logging

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING
if opts.verbose is not None : 
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)

#logging.basicConfig(level=log_level)


#import random
#import sys
#random.seed(365072799328404092)

def canonicalImport(filename):
    preProcess = exampleIO.preProcess

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for (i, row) in enumerate(reader):
            clean_row = [(k, preProcess(v)) for (k, v) in
                         viewitems(row)]
            data_d[i] = dedupe.core.frozendict(clean_row)

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


settings_file = 'canonical_learned_settings.json'
raw_data = 'tests/datasets/restaurant-nophone-training.csv'

data_d, header = canonicalImport(raw_data)

training_pairs = dedupe.trainingDataDedupe(data_d, 
                                           'unique_id', 
                                           5000)

duplicates_s = set(frozenset(pair) for pair in training_pairs['match'])

t0 = time.time()

print('number of known duplicate pairs', len(duplicates_s))

if os.path.exists(settings_file):
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f, 1)
else:
    fields = [{'field' : 'name', 'type': 'String'},
              {'field' : 'name', 'type': 'Exact'},
              {'field' : 'address', 'type': 'String'},
              {'field' : 'cuisine', 'type': 'ShortString'},
              {'field' : 'city', 'type' : 'ShortString'}
              ]

    deduper = dedupe.Dedupe(fields, num_cores=5)
    deduper.sample(data_d, 10000)
    deduper.markPairs(training_pairs)
    deduper.train()
    with open(settings_file, 'wb') as f:
        deduper.writeSettings(f)


alpha = deduper.threshold(data_d, 1.5)

# print candidates
print('clustering...')
clustered_dupes = deduper.match(data_d, threshold=alpha)

print('Evaluate Clustering')
confirm_dupes = set([])
for dupes, score in clustered_dupes:
    for pair in combinations(dupes, 2):
        confirm_dupes.add(frozenset((data_d[pair[0]], 
                                     data_d[pair[1]])))

evaluateDuplicates(confirm_dupes, duplicates_s)

print('ran in ', time.time() - t0, 'seconds')
