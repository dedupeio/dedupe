#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

We start with a CSV file containing our messy data. In this example,
it is listings of early childhood education centers in Chicago
compiled from several different sources.

The output will be a CSV with our clustered results.

For larger datasets, see our [mysql_example](http://open-city.github.com/dedupe/doc/mysql_example.html)
"""

import os
import csv
import re
import collections
import logging
import optparse

import AsciiDammit

import dedupe

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. Added
# for convenience.  To enable verbose logging, run `python
# examples/csv_example/csv_example.py -v`

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


# ## Setup

# Switch to our working directory and set up our input and out put paths,
# as well as our settings and training file locations
os.chdir('./examples/patent_example/')
input_file = 'patent_data_example.csv'
output_file = 'patent_example_output.csv'
settings_file = 'patent_example_learned_settings.json'
training_file = 'patent_example_training.json'


def preProcess(column):
    """
    Do a little bit of data cleaning with the help of
    [AsciiDammit](https://github.com/tnajdek/ASCII--Dammit) and
    Regex. Things like casing, extra spaces, quotes and new lines can
    be ignored.
    """

    column = AsciiDammit.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column


def readData(filename, set_delim='**'):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID and each value is a 
    [frozendict](http://code.activestate.com/recipes/414283-frozen-dictionaries/) 
    (hashable dictionary) of the row fields.

    Remap columns for the following cases:
    - Lat and Long are mapped into a single LatLong tuple
    - Class and Coauthor are stored as delimited strings but mapped into sets

    **Currently, dedupe depends upon records' unique ids being integers
    with no integers skipped. The smallest valued unique id must be 0 or
    1. Expect this requirement will likely be relaxed in the future.**
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            for k in row:
                row[k] = preProcess(row[k])
            row['LatLong'] = (float(row['Lat']), float(row['Lng']))
            del row['Lat']
            del row['Lng']
            row['Class'] = frozenset(row['Class'].split(set_delim))
            row['Coauthor'] = frozenset([author for author
                                         in row['Coauthor'].split(set_delim)
                                         if author != 'none'])
                
            clean_row = [(k, v) for (k, v) in row.items()]
            
            data_d[idx] = dedupe.core.frozendict(clean_row)
            
    return data_d


print 'importing data ...'
data_d = readData(input_file)

## Build the comparators
coauthors = [row['Coauthor'] for idx, row in data_d.items()]
classes = [row['Class'] for idx, row in data_d.items()]
class_comparator = dedupe.distance.cosine.CosineSimilarity(classes)
coauthor_comparator = dedupe.distance.cosine.CosineSimilarity(coauthors)

# ## Training

if os.path.exists(settings_file):
    print 'reading from', settings_file
    deduper = dedupe.Dedupe(settings_file)

else:
    # To train dedupe, we feed it a random sample of records.
    data_sample = dedupe.dataSample(data_d, 150000)
    # Define the fields dedupe will pay attention to
    fields = {
        'Name': {'type': 'String', 'Has Missing':True},
        'LatLong': {'type': 'LatLong', 'Has Missing':True},
        'Class': {'type': 'Custom', 'comparator':class_comparator},
        'Coauthor': {'type': 'Custom', 'comparator': coauthor_comparator},
        }

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # If we have training data saved from a previous run of dedupe,
    # look for it an load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    ## The json file is of the form:
    ## {0: [[{field:val dict of record 1}, {field:val dict of record 2}], ...(more nonmatch pairs)]
    ##  1: [[{field:val dict of record 1}, {field_val dict of record 2}], ...(more match pairs)]
    ## }
    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_sample, training_file)

    # ## Active learning

    # Starts the training loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print 'starting active labeling...'
    deduper.train(data_sample, dedupe.training.consoleLabel)

    # When finished, save our training away to disk
    deduper.writeTraining(training_file)

# ## Blocking


deduper.blocker_types.update({'Custom': (dedupe.predicates.wholeSetPredicate,
                                      dedupe.predicates.commonSetElementPredicate),
                              'LatLong' : (dedupe.predicates.latLongGridPredicate,)})



print 'blocking...'
# Initialize our blocker, which determines our field weights and blocking 
# predicates based on our training data
blocker = deduper.blockingFunction()

# Save our weights and predicates to disk.
# If the settings file exists, we will skip all the training and learning
deduper.writeSettings(settings_file)

## Generate the tfidf canopy as needed

blocker.tfIdfBlocks(full_data)

# Load all the original data in to memory and place
# them in to blocks. Each record can be blocked in many ways, so for
# larger data, memory will be a limiting factor.

blocked_data = dedupe.blockData(data_d, blocker)

# Satore all of our blocked data in to memory
blocked_data = tuple(blocked_data)

# ## Clustering

# Find the threshold that will maximize a weighted average of our precision and recall. 
# When we set the recall weight to 2, we are saying we care twice as much
# about recall as we do precision.
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.goodThreshold(blocked_data, recall_weight=1)

# `duplicateClusters` will return sets of record IDs that dedupe
# believes are all referring to the same entity.

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, threshold)

print '# duplicate sets', len(clustered_dupes)

# ## Writing Results

# Write our original data back out to a CSV with a new column called 
# 'Cluster ID' which indicates which records refer to each other.

cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        cluster_membership[record_id] = cluster_id


with open(output_file, 'w') as f:
    writer = csv.writer(f)

    with open(input_file) as f_input :
        reader = csv.reader(f_input)

        heading_row = reader.next()
        heading_row.insert(0, 'Cluster ID')
        writer.writerow(heading_row)

        for idx, row in enumerate(reader):
            row_id = int(idx)
            cluster_id = cluster_membership[row_id]
            row.insert(0, cluster_id)
            writer.writerow(row)
