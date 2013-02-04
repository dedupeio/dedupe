#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
import time
from collections import defaultdict
import itertools

donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"


def get_sample(cur, size):
  select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"
  cur.execute(select + " ORDER BY RANDOM() LIMIT ?", (size,))
  return dict([(row['donor_id'], row) for row in cur])


settings_file = 'sqlite_example_settings.json'
training_file = 'sqlite_example_training.json'

t0 = time.time()

try:
  os.remove('examples/sqlite_example/blocking_map.db')
except OSError:
  pass


with sqlite3.connect("examples/sqlite_example/blocking_map.db") as con_blocking :

  print 'creating blocking_map database'
  con_blocking.execute("CREATE TABLE blocking_map "
                       "(key TEXT, donor_id INT)")
  con_blocking.commit()


con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = sqlite3.Row
con.execute("ATTACH DATABASE 'examples/sqlite_example/blocking_map.db' AS bm")
con.execute("PRAGMA cache_size = 2000")
con.execute("PRAGMA temp_store = 2")
con.execute("PRAGMA synchronous = OFF")
cur = con.cursor()

print 'selecting random sample from donors table...'
data_d = {}
key_groups = []
num_sample_buckets = 3
for i in range(num_sample_buckets):
  data_sample = get_sample(cur, 700)
  key_groups.append(data_sample.keys())
  data_d.update(data_sample)

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

    if os.path.exists(training_file):
        # read in training json file
        print 'reading labeled examples from ', training_file
        deduper.initializeTraining(training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # get user input for active learning
    deduper.train(data_d, dedupe.training_sample.consoleLabel, key_groups)
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


with sqlite3.connect("examples/sqlite_example/blocking_map.db") as con_blocking :
  print 'creating blocking_map index', time.time() - t0, 'seconds'
  con_blocking.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (key)")
  con_blocking.commit()
  print 'created', time.time() - t0, 'seconds'


  print 'writing largest blocks to file'

  with open('sqlite_example_block_sizes.txt', 'a') as f:
    con.row_factory = None
    f.write(time.asctime())
    f.write('\n')
    for row in con_blocking.execute("SELECT key, COUNT(donor_id) AS block_size "
                                    "FROM blocking_map GROUP BY key "
                                    "ORDER BY block_size DESC LIMIT 10") :

      print row
      f.write(str(row))
      f.write('\n')



print 'ran in', time.time() - t0, 'seconds'
