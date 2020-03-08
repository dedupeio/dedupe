import itertools
import csv
import os
import time
import optparse
import logging

import dedupe

import exampleIO


def canonicalImport(filename):
    preProcess = exampleIO.preProcess
    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = {k: preProcess(v) for (k, v) in
                         row.items()}
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


if __name__ == '__main__':

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

    settings_file = 'canonical_data_matching_learned_settings'

    data_1, header = canonicalImport('tests/datasets/restaurant-1.csv')
    data_2, _ = canonicalImport('tests/datasets/restaurant-2.csv')

    training_pairs = dedupe.training_data_link(data_1, data_2, 'unique_id', 5000)

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
        deduper.prepare_training(data_1, data_2, sample_size=10000)
        deduper.mark_pairs(training_pairs)
        deduper.train()

        with open(settings_file, 'wb') as f:
            deduper.write_settings(f)

    # print candidates
    print('clustering...')
    clustered_dupes = deduper.join(data_1, data_2, threshold=0.5)

    print('Evaluate Clustering')
    confirm_dupes = set(frozenset(pair)
                        for pair, score in clustered_dupes)

    evaluateDuplicates(confirm_dupes, duplicates_s)

    print('ran in ', time.time() - t0, 'seconds')

    # print candidates
    print('clustering...')
    clustered_dupes = deduper.join(data_1, data_2, threshold=0.5, constraint='many-to-one')

    print('Evaluate Clustering')
    confirm_dupes = set(frozenset(pair)
                        for pair, score in clustered_dupes)

    evaluateDuplicates(confirm_dupes, duplicates_s)

    print('ran in ', time.time() - t0, 'seconds')
