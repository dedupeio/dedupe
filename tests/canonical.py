#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from future.utils import viewitems

from itertools import combinations, groupby
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
if opts.verbose is not None:
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)


def canonicalImport(filename):
    preProcess = exampleIO.preProcess

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for (i, row) in enumerate(reader):
            clean_row = {k: preProcess(v) for (k, v) in
                         viewitems(row)}
            data_d[i] = clean_row

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


settings_file = 'canonical_learned_settings'
raw_data = 'tests/datasets/restaurant-nophone-training.csv'

data_d, header = canonicalImport(raw_data)

training_pairs = dedupe.trainingDataDedupe(data_d,
                                           'unique_id',
                                           5000)

duplicates = set()
for _, pair in groupby(sorted(data_d.items(),
                              key=lambda x: x[1]['unique_id']),
                       key=lambda x: x[1]['unique_id']):
    pair = list(pair)
    if len(pair) == 2:
        a, b = pair
        duplicates.add(frozenset((a[0], b[0])))

t0 = time.time()

print('number of known duplicate pairs', len(duplicates))

if os.path.exists(settings_file):
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f, 1)

else:
    fields = [{'field': 'name', 'type': 'String'},
              {'field': 'name', 'type': 'Exact'},
              {'field': 'address', 'type': 'String'},
              {'field': 'cuisine', 'type': 'ShortString',
               'has missing': True},
              {'field': 'city', 'type': 'ShortString'}
              ]

    deduper = dedupe.Dedupe(fields, num_cores=5)
    deduper.sample(data_d, 10000)
    deduper.markPairs(training_pairs)
    deduper.train(index_predicates=False)
    with open(settings_file, 'wb') as f:
        deduper.writeSettings(f)


alpha = deduper.threshold(data_d, 1)

# print candidates
print('clustering...')
clustered_dupes = deduper.match(data_d, threshold=alpha)


print('Evaluate Clustering')
confirm_dupes = set([])
for dupes, score in clustered_dupes:
    for pair in combinations(dupes, 2):
        confirm_dupes.add(frozenset(pair))

evaluateDuplicates(confirm_dupes, duplicates)

print('ran in ', time.time() - t0, 'seconds')
