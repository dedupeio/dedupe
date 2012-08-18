from itertools import combinations
import csv
import exampleIO
import dedupe
import os
import time

def canonicalImport(filename) :
    preProcess = exampleIO.preProcess

    data_d = {}
    clusters = {}
    duplicates = set([])

    with open(filename) as f :
        reader = csv.DictReader(f)
        for i, row in enumerate(reader) :
            clean_row = [(k, preProcess(v)) for k,v in row.iteritems()]
            data_d[i] = dedupe.core.frozendict(clean_row)
            clusters.setdefault(row['unique_id'], []).append(i)
            

    for unique_id, cluster in clusters.iteritems() :
      if len(cluster) > 1 :
        for pair in combinations(cluster, 2) :
          duplicates.add(frozenset(pair))

    return(data_d, reader.fieldnames, duplicates)

def evaluateDuplicates(found_dupes, true_dupes) :
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)
    uncovered_dupes = true_dupes.difference(found_dupes)

    print "found duplicate"
    print len(found_dupes)

    print "precision"
    print 1 - len(false_positives)/float(len(found_dupes))

    print "recall"
    print len(true_positives)/float(len(true_dupes))

    #eturn uncovered_dupes, false_positives

def printPairs(pairs) :
    for pair in pairs :
        print ""
        for instance in tuple(pair) :
            print data_d[instance].values()


settings_file = 'restaurant_learned_settings.json'
num_training_dupes = 200
num_training_distinct = 1600
num_iterations = 10

data_d, header, duplicates_s = canonicalImport("examples/datasets/restaurant-nophone-training.csv")

t0 = time.time()

print "number of duplicates pairs"
print len(duplicates_s)
print ""

if os.path.exists(settings_file) :
    deduper = dedupe.Dedupe(settings_file, 'settings_file')
else :
    fields = {'name' : {'type': 'String'}, 
          'address' : {'type' :'String'},
          'city' : {'type': 'String'},
          'cuisine' : {'type': 'String'},
          'name:city' : {'type': 'Interaction',
                         'interaction-terms': ['name', 'city']}
          }
    deduper = dedupe.Dedupe(fields, 'fields')
    
    deduper.training_pairs = dedupe.training_sample.randomTrainingPairs(data_d,
                                                                       duplicates_s,
                                                                       num_training_dupes,
                                                                       num_training_distinct)

    deduper.trainingDistance()
    deduper.train(num_iterations)
    
deduper.findDuplicates(data_d, threshold=0.5)

print "Evaluate Scoring"
found_dupes = set([frozenset(dupe_pair[0])
                   for dupe_pair in deduper.dupes])

evaluateDuplicates(found_dupes, duplicates_s)

print "Evaluate Clustering"

clustered_dupes = deduper.duplicateClusters(.5)

confirm_dupes = set([])
for dupe_set in clustered_dupes :
    if (len(dupe_set) == 2) :
        confirm_dupes.add(frozenset(dupe_set))
    else :
        for pair in combinations(dupe_set, 2) :
            confirm_dupes.add(frozenset(pair))



evaluateDuplicates(confirm_dupes, duplicates_s)

print "ran in ", time.time() - t0, "seconds"
