"""
This is an example of working with very large data. There are about 700,000 unduplicated
donors in this database of Illinois political campaign contributions.

While we might be able to keep these donor records in memory, we cannot possibly store all 
the comparison pairs we will make. 

Because of performance issues that we are still working through, this example is broken into
two files, sqlite_blocking.py which blocks the data, sqlite_clustering.py which clusters the
blocked data.
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
from collections import defaultdict
import itertools
import time
import os

os.chdir('./examples/sqlite_example/')
settings_file = 'sqlite_example_settings.json'
training_file = 'sqlite_example_training.json'

# When we compare records, we don't care about differences in case. 
# Lowering the case in SQL is much faster than in Python.
donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"


def get_sample(cur, size):
  """
  Returns a random sample of donors of size=size
  """
  cur.execute(donor_select + " ORDER BY RANDOM() LIMIT ?", (size,))
  return dict([(row['donor_id'], row) for row in cur])

t0 = time.time()

# For performance reasons, its faster to delete and recreate a large sqlite database file than to delete a large table.
try:
  os.remove('blocking_map.db')
except OSError:
  pass

# Create our blocking map table in a separate database.
with sqlite3.connect("blocking_map.db") as con_blocking :

  print 'creating blocking_map database'
  con_blocking.execute("CREATE TABLE blocking_map "
                       "(key TEXT, donor_id INT)")
  con_blocking.commit()

con = sqlite3.connect("illinois_contributions.db")
con.row_factory = sqlite3.Row
# Attach blocking_map to our primary database
con.execute("ATTACH DATABASE 'blocking_map.db' AS bm")
# To help with such large scale data, we are increasing the sqlite cache size. 
con.execute("PRAGMA cache_size = 2000")
cur = con.cursor()

# Unlike csv_example.py, we select from the database to get a random sample for training. As the dataset grows, duplicate pairs become more rare. To account for this, we are taking a larger sample (3x 700 records) for training.
print 'selecting random sample from donors table...'
data_samples = []
num_sample_buckets = 3
for i in range(num_sample_buckets):
  data_sample = get_sample(cur, 700)
  data_samples.append(data_sample)

if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
    fields = {'first_name': {'type': 'String'},
              'last_name': {'type': 'String'},
              'address_1': {'type': 'String'},
              'address_2': {'type': 'String'},
              'city': {'type': 'String'},
              'state': {'type': 'String'},
              'zip': {'type': 'String'},
              }
    deduper = dedupe.Dedupe(fields)

    # Sometimes we will want to add additional labeled examples to a training file. To do this
    # We load the training file with initializeTraining
    if os.path.exists(training_file):
        # read in training json file
        print 'reading labeled examples from ', training_file
        deduper.initializeTraining(training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # get user input for active learning
    deduper.train(data_samples, dedupe.training_sample.consoleLabel)
    deduper.writeTraining(training_file)

print 'blocking...'
t_block = time.time()
blocker = deduper.blockingFunction(eta=0.001, epsilon=5)

deduper.writeSettings(settings_file)
print 'blocked in', time.time() - t_block, 'seconds'

print 'creating inverted index'
full_data = ((row['donor_id'], row) for row in con.execute(donor_select))
blocker.invertIndex(full_data)

# print 'token vector', blocker.token_vector
# print 'inverted index', blocker.inverted_index

print 'creating TF/IDF canopies'
blocker.canopies = {}
counter = 1

# pure hackery that we need to fix in blocker
seen_preds = set([])
tfidf_thresholds = []
for threshold, field in blocker.tfidf_thresholds :
  if (threshold.threshold, field) not in seen_preds :
    tfidf_thresholds.append((threshold, field))
    seen_preds.add((threshold.threshold, field))
    

for threshold, field in tfidf_thresholds :
    print (str(counter) + "/" + str(len(tfidf_thresholds))), threshold.threshold, field
    canopy = blocker.createCanopies(field, threshold)
    blocker.canopies[threshold.__name__ + field] = canopy
    counter += 1

print 'created canopies at', time.time() - t0, 'seconds'

del blocker.inverted_index
del blocker.token_vector

print 'writing blocking map'
def block_data() :
    full_data = ((row['donor_id'], row) for row in con.execute(donor_select))
    for i, (donor_id, record) in enumerate(full_data) :
        if i % 10000 == 0 :
            print i, ',', time.time() - t0, 'seconds'
        # should move this set code into blocker
        for key in set(str(block_key) for block_key in blocker((donor_id, record))):
            yield (key, donor_id)


con.executemany("INSERT INTO bm.blocking_map VALUES (?, ?)",
                block_data())

con.commit()
cur.close()
con.close()


with sqlite3.connect("blocking_map.db") as con_blocking :
  print 'creating blocking_map index', time.time() - t0, 'seconds'
  con_blocking.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (key)")
  con_blocking.commit()
  print 'created', time.time() - t0, 'seconds'

print 'ran in', time.time() - t0, 'seconds'
