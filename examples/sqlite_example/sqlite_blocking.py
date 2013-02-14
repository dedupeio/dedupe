#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is an example of working with very large data. There are about
700,000 unduplicated donors in this database of Illinois political
campaign contributions.

While we might be able to keep these donor records in memory, we
cannot possibly store all the comparison pairs we will make.

Because of performance issues that we are still working through, this
example is broken into two files, sqlite_blocking.py which blocks the
data, sqlite_clustering.py which clusters the blocked data.
"""
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


def getSample(con, size):
  """
  Returns a random sample of pairs of donors of a given size
  """

  dim = con.execute("SELECT MAX(donor_id) FROM donors").next()[0]

  random_pairs = dedupe.randomPairs(dim, size, zero_indexed=False)

  all_ids = [str(record_id) for pair in random_pairs for record_id in pair]

  temp_d = {}

  for row in con.execute(donor_select + " WHERE donor_id IN (%s)" % ','.join(
    '?'*size*2), all_ids) :
    temp_d[row['donor_id']] = row

  return tuple((((record_id_1, temp_d[record_id_1]),
                 (record_id_2, temp_d[record_id_2]))
                for record_id_1, record_id_2
                in random_pairs))



t0 = time.time()

# For performance reasons, its faster to delete and recreate a large sqlite database file than to delete a large table.
try:
  os.remove('blocking_map.db')
except OSError:
  pass

# Create our blocking map table in a separate database.

print 'creating blocking_map database'
with sqlite3.connect("blocking_map.db") as con_blocking :
  con_blocking.execute("CREATE TABLE blocking_map "
                       "(key TEXT, donor_id INT)")
  con_blocking.commit()

con = sqlite3.connect("illinois_contributions.db")
# This is a nice way to get records we can index by the name of the
# field
con.row_factory = sqlite3.Row

# Attach blocking_map to our primary database
con.execute("ATTACH DATABASE 'blocking_map.db' AS bm")

# To help with such large scale data, we are increasing the sqlite
# cache size.
con.execute("PRAGMA cache_size = 2000")



# Unlike csv_example.py, we select from the database to get a random
# sample for training. As the dataset grows, duplicate pairs become
# more rare. We need positive examples to train dedupe, so we have to
# signficantly increase the size of the sample
print 'selecting random sample from donors table...'
data_sample = getSample(con, 750000)

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

    # Sometimes we will want to add additional labeled examples to a
    # training file. To do this can just load the existing labeled
    # pairs...
    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_samples, training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # ... and then call training with our interactive function
    deduper.train(data_samples, dedupe.training_sample.consoleLabel)
    deduper.writeTraining(training_file)

print 'blocking...'
t_block = time.time()
blocker = deduper.blockingFunction(eta=0.001, epsilon=5)
deduper.writeSettings(settings_file)
print 'blocked in', time.time() - t_block, 'seconds'

# So the learning is done and we have our blocker. However we cannot
# block the data in memory. We have to pass through all the data and
# create a blocking map table.
#
# First though, if we learned a tf-idf predicate, we have to create an
# tfIDF blocks for the full data set.
print 'creating inverted index'
full_data = ((row['donor_id'], row) for row in con.execute(donor_select))
blocker.tfIdfBlocks(full_data)


# Finally, we are ready to block the data. We'll do this by creating
# a generator that yields a (block_key, donor_id) tuples. We guarantee
# that these tuples will be unique
print 'writing blocking map'
def block_data() :
    full_data = ((row['donor_id'], row) for row in con.execute(donor_select))
    for i, (donor_id, record) in enumerate(full_data) :
        if i % 10000 == 0 :
            print i, ',', time.time() - t0, 'seconds'
        for key in blocker((donor_id, record)) :
            yield (key, donor_id)

# This takes the generator and writes into into our table
con.executemany("INSERT INTO bm.blocking_map VALUES (?, ?)",
                block_data())
con.commit()
con.close()

# Finally, we create an index on the blocking_key so that the group by
# queries we will be making in sqlite_clustering can happen in a
# reasonable time
with sqlite3.connect("blocking_map.db") as con_blocking :
  print 'creating blocking_map index', time.time() - t0, 'seconds'
  con_blocking.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (key)")
  con_blocking.commit()
  print 'created', time.time() - t0, 'seconds'

print 'ran in', time.time() - t0, 'seconds'
