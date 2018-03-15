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
import optparse
import logging

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING
if opts.verbose:
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
            clean_row = {k: preProcess(v) for (k, v) in
                         viewitems(row)}
            data_d[filename + str(i)] = clean_row

    return data_d, reader.fieldnames


def evaluateDuplicates(found_dupes, true_dupes):
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)

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

all_data = data_1.copy()
all_data.update(data_2)

duplicates_s = set()
for _, pair in itertools.groupby(sorted(all_data.items(),
                                        key=lambda x: x[1]['unique_id']),
                                 key=lambda x: x[1]['unique_id']):
    pair = list(pair)
    if len(pair) == 2:
        a, b = pair
        duplicates_s.add(frozenset((a[0], b[0])))

t0 = time.time()

print('number of known duplicate pairs', len(duplicates_s))

if os.path.exists(settings_file):
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticRecordLink(f)
else:
    fields = [{'field': 'name', 'type': 'String'},
              {'field': 'address', 'type': 'String'},
              {'field': 'cuisine', 'type': 'String'},
              {'field': 'city', 'type': 'String'}
              ]

    deduper = dedupe.RecordLink(fields)
    deduper.sample(data_1, data_2, 10000)
    deduper.markPairs(training_pairs)
    deduper.train()

alpha = deduper.threshold(data_1, data_2)

with open(settings_file, 'wb') as f:
    deduper.writeSettings(f, index=True)


# print candidates
print('clustering...')
clustered_dupes = deduper.match(data_1, data_2, threshold=alpha)

print('Evaluate Clustering')
confirm_dupes = set(frozenset(pair)
                    for pair, score in clustered_dupes)

evaluateDuplicates(confirm_dupes, duplicates_s)

print('ran in ', time.time() - t0, 'seconds')
