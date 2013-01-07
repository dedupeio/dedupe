#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
import time
from collections import defaultdict
import itertools

def dict_factory(cursor, row):
    d = {}
    donor_id = 0
    for idx, col in enumerate(cursor.description):
      if col[0] == 'donor_id' :
        donor_id = row[idx]
      else :
        column = re.sub('  +', ' ', row[idx])
        column = re.sub('\n', ' ', column)
        column = column.strip().strip('"').strip("'").lower()
        d[col[0]] = column
    return (donor_id, d)

def blocking_factory(cursor, row):
    d = {}
    donor_id = 0
    key = ''
    for idx, col in enumerate(cursor.description):
      if col[0] == 'donor_id' :
        donor_id = row[idx]
      elif col[0] == 'key' :
        key = row[idx]
      else :
        column = re.sub('  +', ' ', row[idx])
        column = re.sub('\n', ' ', column)
        column = column.strip().strip('"').strip("'").lower()
        d[col[0]] = column
    return (key, (donor_id, d))

def get_sample(cur, size):
  cur.execute("SELECT * FROM donors ORDER BY RANDOM() LIMIT ?", (size,))
  return dict([(row['donor_id'], row) for row in cur])


settings_file = 'sqlite_example_settings.json'
training_file = 'sqlite_example_training.json'

t0 = time.time()

print 'selecting random sample from donors table...'
con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = sqlite3.Row
cur = con.cursor()

try:
  os.remove('examples/sqlite_example/blocking_map.db')
except:
  raise

con_blocking = sqlite3.connect("examples/sqlite_example/blocking_map.db")
cur_blocking = con_blocking.cursor()

print 'creating blocking_map database'
cur_blocking.execute("CREATE TABLE blocking_map "
            "(key TEXT, donor_id INT, PRIMARY KEY(key,donor_id))")
cur_blocking.execute("CREATE INDEX key_index ON blocking_map (key)")
cur_blocking.execute("CREATE INDEX donor_id_index ON blocking_map (donor_id)")
cur_blocking.execute("CREATE INDEX itx_index ON blocking_map (key, donor_id)")

cur.execute("ATTACH DATABASE 'blocking_map.db' AS blocking_map;")

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

def createSelector(field, con) :
    cur = con.cursor()

    def selector(doc_ids) :

      doc_ids = ', '.join([str(doc_id) for doc_id in doc_ids ])
      sql = "SELECT donor_id, %s, address_1, address_2, last_name FROM donors WHERE donor_id IN (%s)" % (field, doc_ids)
      #print sql
      for row in cur.execute(sql) :
        #print row
        yield (row['donor_id'], row[str(field)])

    return selector


print 'creating inverted index'
full_data = ((row['donor_id'], row) for row in con.execute("SELECT * FROM donors LIMIT 100000"))
blocker.invertIndex(full_data)

# print 'token vector', blocker.token_vector
# print 'inverted index', blocker.inverted_index

print 'creating canopies'
blocker.canopies = {}
counter = 1
for threshold, field in blocker.tfidf_thresholds :
    print (str(counter) + "/" + str(len(blocker.tfidf_thresholds))), threshold.threshold, field
    # selector = createSelector(field, con)
    canopy = blocker.createCanopies(field, threshold)
    blocker.canopies[threshold.__name__ + field] = canopy
    counter += 1

del blocker.inverted_index
del blocker.token_vector

print 'writing blocking map'
def block_data() :
    full_data = ((row['donor_id'], row) for row in con.execute("SELECT * FROM donors LIMIT 100000"))
    for donor_id, record in full_data :
        if donor_id % 10000 == 0 :
            print donor_id
        for key in blocker((donor_id, record)):
            yield (str(key), donor_id)


con.executemany("INSERT OR IGNORE INTO blocking_map VALUES (?, ?)",
                block_data())

con.commit()

print 'writing largest blocks to file'

with open('sqlite_example_block_sizes.txt', 'a') as f:
    con.row_factory = None
    f.write(time.asctime())
    f.write('\n')
    for row in con.execute("SELECT key, COUNT(donor_id) AS block_size "
                           "FROM blocking_map GROUP BY key "
                           "ORDER BY block_size DESC LIMIT 10") :

        print row
        f.write(str(row))
        f.write('\n')

print 'reading blocked data'
cur = con.cursor()


cur.execute('select * from donors join '
 '(select key, donor_id from blocking_map '
 'join (select key, count(donor_id) num_candidates from blocking_map '
 'group by key having num_candidates > 1) '
 'as bucket using (key)) as candidates using (donor_id)')

block_keys = (row['key'] for row in con.execute('select key, count(donor_id) as num_candidates from blocking_map group by key having num_candidates > 1'))


def candidates_gen() :
    candidate_set = set([])
    for block_key in block_keys :
        block = set(itertools.combinations((row['donor_id'] for row in con.execute('select * from donors inner join blocking_map using (donor_id) where key = ? order by donor_id', (block_key,))), 2))
        new = block - candidate_set
        candidate_set |= new
        for candidate_pair in new :
            yield candidate_pair
    
print 'clustering...'
clustered_dupes = deduper.duplicateClustersII(candidates_gen())

print '# duplicate sets'
print len(clustered_dupes)

cur.close()
con.close()
print 'ran in', time.time() - t0, 'seconds'
