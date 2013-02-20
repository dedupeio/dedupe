#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The following code demonstrates how to use dedupe with a flat (CSV)
file. All operations are performed in memory, so it won't work for
data sets that are larger than ~10,000 rows.
"""

import os
import csv
import re
import collections
import AsciiDammit
import logging
import optparse

import dedupe

# We will setup csv_example.py so that the user can run in a more
# verbose mode by calling `python examples/csv_example/csv_example.py
# -v`
#
# This optparse and logging optparse business is not necessary for
# dedupe

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


# ## Setting up the Data

# We start with a csv file with our messy data. In this example, it is
# listings of early childhood education centers in Chicago compiled
# from several different sources. The output file will also be a csv,
# but with our clustered results.

os.chdir('./examples/csv_example/')
input_file = 'csv_example_messy_input.csv'
output_file = 'csv_example_output.csv'

# We can save a settings file to initialize a dedupe instance without
# having to retrain

settings_file = 'csv_example_learned_settings.json'

# We can also save the examples the users labels if we want to add on
# to them later for more training

training_file = 'csv_example_training.json'


def preProcess(column):
    """
    Our goal here is to find meaningful duplicates, so things like
    casing, extra spaces, quotes and new lines can be
    ignored. `preProcess` removes these.
    """

    column = AsciiDammit.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column


def readData(filename):
    """
    We read in the data from a CSV file and as we do, we `preProcess`
    it. The output is a dictionary of records, where the key is a
    unique record ID and each value is a `frozendict` (basically a
    hashable dictionary) of the row fields.

    **Currently, dedupe depends upon records' unique ids being integers
    with no integers skipped. The smallest valued unique id must be 0 or
    1.**

    **This all is due to how we generate random samples of pairs, and this
    requirement will likely be relaxed in the future.**
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in
                         row.iteritems()]
            row_id = int(row['Id'])
            data_d[row_id] = dedupe.core.frozendict(clean_row)

    return (data_d, reader.fieldnames)


print 'importing data ...'
(data_d, header) = readData(input_file)

# ## Teaching Dedupe to Compare Records

# ### Using what we learned before
# If the settings files, which we mentioned above, exists, then we
# read it in. Passing in a settings file is one of the three ways to
# initialize a dedupe instance. We won't need to do any learning.

if os.path.exists(settings_file):
    print 'reading from', settings_file
    deduper = dedupe.Dedupe(settings_file)

# ### Learning anew
else:

    
    # In order to train dedupe, we need to compare some records. We can't
    # compare them all, because the number of possible combinations can be
    # much too large (~0.5*N^2). We take a random sample of all possible
    # pairs.

    data_sample = dedupe.dataSample(data_d, 150000)

    # We can initialize a dedupe instance by declaring a field
    # definition for the data. This defines the fields that we want to
    # compare and what type of field it is so that dedupe know how to
    # compare them.

    fields = {
        'Site name': {'type': 'String'},
        'Address': {'type': 'String'},
        'Zip': {'type': 'String'},
        'Phone': {'type': 'String'},
        }
    deduper = dedupe.Dedupe(fields)

    # Dedupe will learn to predict if two records are duplicates based
    # upon their similarity. In this example, that similarity is a
    # weighted combination of the field by field [string
    # similarity](http://en.wikipedia.org/wiki/String_metric) between
    # records. Dedupe learns these weights.

    # #### Learning from saved, labeled examples

    # Dedupe will ask a user to label pairs of records as duplicates or
    # not. These labeled records can be saved and reused for later
    # training. To train dedupe with these examples, call `deduper.train`
    # as shown.

    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_sample, training_file)

    # #### Actively learning

    # Dedupe can actively learn, that means it will select the records
    # it is most uncertain about and will ask the user to label it. It
    # will then learn from that labeling, update, and ask for the next
    # most uncertain pair.

    # To do this, train method requires that you pass it a function to
    # do this labeling, in this case, `consoleLabel`.

    # For `consoleLabel`, use 'y', 'n' and 'u' keys to flag duplicates,
    # 'f' when you are finished.

    print 'starting active labeling...'
    deduper.train(data_sample, dedupe.training.consoleLabel)

    # Save away our labeled training pairs to a JSON file.

    deduper.writeTraining(training_file)

# ## Teaching Dedupe How to Block Records

# Now that dedupe knows how to compare records, we use the same
# training data to block records in to groups. The goal is to reduce
# the total number of comparisons.

# `blockingFunction` learns the blocking rules, if not already defined
# in `settings_file` and returns a function. That function will take a
# record and return all the blocks it will fit in to.

print 'blocking...'
blocker = deduper.blockingFunction()

# Save our settings file, which includes learned weights and blocking
# rules.

deduper.writeSettings(settings_file)

# ## Blocking the Data

# `blockingIndex` loads all the original data in to memory and places
# them in to blocks. Each record can be blocked in many ways, so for
# larger data, memory will be a limiting factor.

blocked_data = dedupe.blockData(data_d, blocker)

# `dedupe.blockData` returns a generator, because we typically would
# not want to keep all the pairs we want to compare in memory. In this
# example though, the data is small enough that we can handle it. We
# store the block of data in a tuple, which we'll use as a sample to
# learn a good threshold

blocked_data = tuple(blocked_data)

# ## Identifying Duplicates

# There is always a tradeoff between precision and recall. This function
# tries to find the threshold that will maximize a weighted average of both.
# When we set the recall weight to 2, we are saying we care twice as much
# about recall as we do precision.
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.goodThreshold(blocked_data, recall_weight=2)

# `duplicateClusters` will return sets of record IDs that dedupe
# believes are all referring to the same entity.

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, threshold)

print '# duplicate sets', len(clustered_dupes)

# ## Writing Results

# Now that we have our clustered duplicates, we write our original
# data back out to a CSV with a new column called 'Cluster ID' which
# indicates which records refer to each other.

cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        cluster_membership[record_id] = cluster_id


with open(output_file, 'w') as f:
    writer = csv.writer(f)
    heading_row = header
    heading_row.insert(0, 'Cluster ID')
    writer.writerow(heading_row)

    with open(input_file) as f_input :
        reader = csv.reader(f_input)
        reader.next()

        for row in reader:
            row_id = int(row[0])
            cluster_id = cluster_membership[row_id]
            row.insert(0, cluster_id)
            writer.writerow(row)


