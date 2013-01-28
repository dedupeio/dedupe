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
  cur.execute(donor_select + " ORDER BY RANDOM() LIMIT ?", (size,))
  return dict([(row['donor_id'], row) for row in cur])


settings_file = 'sqlite_example_settings.json'
training_file = 'sqlite_example_training.json'

t0 = time.time()



try:
  os.remove('examples/sqlite_example/blocking_map.db')
except OSError:
  pass

con_blocking = sqlite3.connect("examples/sqlite_example/blocking_map.db")
cur_blocking = con_blocking.cursor()

print 'creating blocking_map database'
cur_blocking.execute("CREATE TABLE blocking_map "
            "(key TEXT, donor_id INT, PRIMARY KEY(key,donor_id))")
cur_blocking.execute("CREATE INDEX key_index ON blocking_map (key)")
cur_blocking.execute("CREATE INDEX donor_id_index ON blocking_map (donor_id)")
cur_blocking.execute("CREATE INDEX itx_index ON blocking_map (key, donor_id)")
con_blocking.commit()
cur_blocking.close()
con_blocking.close()



con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = sqlite3.Row
con.execute("ATTACH DATABASE 'examples/sqlite_example/blocking_map.db' AS bm")
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

print 'creating canopies'
blocker.canopies = {}
counter = 1
for threshold, field in blocker.tfidf_thresholds :
    print (str(counter) + "/" + str(len(blocker.tfidf_thresholds))), threshold.threshold, field
    canopy = blocker.createCanopies(field, threshold)
    blocker.canopies[threshold.__name__ + field] = canopy
    counter += 1

del blocker.inverted_index
del blocker.token_vector

print 'writing blocking map'
def block_data() :
    full_data = ((row['donor_id'], row) for row in con.execute(donor_select))
    for donor_id, record in full_data :
        if donor_id % 10000 == 0 :
            print donor_id
        for key in blocker((donor_id, record)):
            yield (str(key), donor_id)


con.executemany("INSERT OR IGNORE INTO bm.blocking_map VALUES (?, ?)",
                block_data())

con.commit()

print 'writing largest blocks to file'

with open('sqlite_example_block_sizes.txt', 'a') as f:
    con.row_factory = None
    f.write(time.asctime())
    f.write('\n')
    for row in con.execute("SELECT key, COUNT(donor_id) AS block_size "
                           "FROM bm.blocking_map GROUP BY key "
                           "ORDER BY block_size DESC LIMIT 10") :

        print row
        f.write(str(row))
        f.write('\n')


cur.close()
con.close()
print 'ran in', time.time() - t0, 'seconds'
