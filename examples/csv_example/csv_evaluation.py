import csv
import collections
import itertools
import os

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


def dupePairs(filename, rowname) :
    dupe_d = collections.defaultdict(list)

    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            dupe_d[row[rowname]].append(row['Id'])

    if 'x' in dupe_d :
        del dupe_d['x']

    dupe_s = set([])
    for (unique_id, cluster) in dupe_d.iteritems():
        if len(cluster) > 1:
            for pair in itertools.combinations(cluster, 2):
                dupe_s.add(frozenset(pair))

    return dupe_s

os.chdir('./examples/csv_example/')

manual_clusters = 'csv_example_input_with_true_ids.csv'
dedupe_clusters = 'csv_example_output.csv'

true_dupes = dupePairs(manual_clusters, 'True Id')
test_dupes = dupePairs(dedupe_clusters, 'Cluster ID')

evaluateDuplicates(test_dupes, true_dupes)

