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
logging.basicConfig(level=log_level)

# create a random set of training pairs based on known duplicates

def randomTrainingPairs(data_d,
                        duplicates_s,
                        n_training_dupes,
                        n_training_distinct,
                        ):

    if n_training_dupes < len(duplicates_s):
        duplicates = random.sample(duplicates_s, n_training_dupes)
    else:
        duplicates = duplicates_s

    duplicates = [(data_d[tuple(pair)[0]], data_d[tuple(pair)[1]])
                  for pair in duplicates]

    all_pairs = list(itertools.combinations(data_d, 2))
    all_nonduplicates = set(all_pairs) - set(duplicates_s)

    nonduplicates = random.sample(all_nonduplicates, n_training_distinct)

    nonduplicates = [(data_d[pair[0]], data_d[pair[1]])
                     for pair in nonduplicates]

    return {'distinct': nonduplicates, 'match': duplicates}


def canonicalImport(filename, base=False):
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


    # eturn uncovered_dupes, false_positives

def printPairs(pairs):
    for pair in pairs:
        print ''
        for instance in tuple(pair):
            print data_d[instance].values()


settings_file = 'canonical_data_matching_learned_settings'
num_training_dupes = 400
num_training_distinct = 2000
num_iterations = 10

data_1, header = canonicalImport('test/datasets/restaurant-1.csv', base=True)
data_2, _ = canonicalImport('test/datasets/restaurant-2.csv', base=False)
data_d = dict(data_1.items() + data_2.items())

clusters_1 = {}
for k, row in data_1.items() :
    clusters_1.setdefault(row['unique_id'], []).append(k)

clusters_2 = defaultdict(list)
for k, row in data_2.items() :
    clusters_2.setdefault(row['unique_id'], []).append(k)

duplicates_s = set([])
for (unique_id, cluster_1) in clusters_1.iteritems():
    cluster_2 = clusters_2[unique_id]
    if cluster_2 :
        for pair in itertools.product(cluster_1, cluster_2):
            duplicates_s.add(frozenset(pair))



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

    data_sample = dedupe.dataSampleRecordLink(data_1,
                                              data_2,
                                              100000) 

    deduper = dedupe.RecordLink(fields, data_sample)
    deduper.num_iterations = num_iterations

    print "Using a random sample of training pairs..."

    deduper.training_pairs = randomTrainingPairs(data_d,
                                                 duplicates_s,
                                                 num_training_dupes,
                                                 num_training_distinct)

    deduper._addTrainingData(deduper.training_pairs)


    deduper.trainClassifier()
    deduper.trainBlocker()

    deduper._logLearnedWeights()

    deduper.writeSettings(settings_file)



print 'blocking...'
blocked_data = tuple(dedupe.blockDataRecordLink(data_1, data_2, deduper.blocker))

alpha = deduper.goodThreshold(blocked_data)


# print candidates
print 'clustering...'
clustered_dupes = deduper.match(blocked_data, threshold=alpha)



print 'Evaluate Scoring'
found_dupes = set([frozenset(pair) for (pair, score) in deduper.matches
                  if score > alpha])

evaluateDuplicates(found_dupes, duplicates_s)

print 'Evaluate Clustering'

confirm_dupes = set([])
for dupe_set in clustered_dupes:
    if len(dupe_set) == 2:
        confirm_dupes.add(frozenset(dupe_set))
    else:
        for pair in itertools.combinations(dupe_set, 2):
            confirm_dupes.add(frozenset(pair))

evaluateDuplicates(confirm_dupes, duplicates_s)

print 'ran in ', time.time() - t0, 'seconds'
