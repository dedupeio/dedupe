#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
import time
from collections import defaultdict

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
  return dict(cur.fetchall())


settings_file = 'sqlite_example_settings.json'
training_file = 'sqlite_example_training.json'

t0 = time.time()

print 'selecting random sample from donors table...'
con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = dict_factory
cur = con.cursor()

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
blocker = deduper.blockingFunction(eta=0.005, epsilon=5)

deduper.writeSettings(settings_file)
print 'blocked in ', time.time() - t_block, 'seconds'

print 'deleting existing blocking map'
cur.execute("DROP TABLE IF EXISTS blocking_map")
print 'creating blocking_map'
cur.execute("CREATE TABLE blocking_map "
            "(key TEXT, donor_id INT, PRIMARY KEY(key,donor_id))")
cur.execute("CREATE INDEX key_index ON blocking_map (key)")
cur.execute("CREATE INDEX donor_id_index ON blocking_map (donor_id)")
cur.execute("CREATE INDEX itx_index ON blocking_map (key, donor_id)")

def createSelector(field, con) :

    cur = con.cursor()
    def selector(doc_id) :
      sql = "SELECT %s FROM donors WHERE donor_id = %s" % (field, doc_id)
      #print sql
      cur.execute(sql)
      field_value = cur.fetchone()
      #print field_value

      return field_value[1][field]

    return selector


print 'creating inverted index'
blocker.invertIndex(con.execute("SELECT * FROM donors limit 10000"))

print 'creating canopies'
blocker.canopies = {}
counter = 1
for threshold, field in blocker.tfidf_thresholds :
    print threshold.threshold, field
    print str(counter) + "/" + str(len(blocker.tfidf_thresholds))
    selector = createSelector(field, con)
    canopy = blocker.createCanopies(selector, field, threshold)
    blocker.canopies[threshold.__name__ + field] = canopy
    counter += 1

print 'writing blocking map'
def block_data() :
    for donor_id, record in con.execute("SELECT * FROM donors limit 10000") :
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
    con.row_factory = dict_factory
    

print 'reading blocked data'
con.row_factory = blocking_factory
cur = con.cursor()
cur.execute('select * from donors join '
  '(select key, donor_id from blocking_map '
  'join (select key, count(donor_id) num_candidates from blocking_map '
  'group by key having num_candidates > 1) '
  'as bucket using (key)) as candidates using (donor_id)')
blocked_data = defaultdict(list)
for k, v in cur :
    blocked_data[k].append(v)

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data)

print '# duplicate sets'
print len(clustered_dupes)

cur.close()
con.close()
print 'ran in ', time.time() - t0, 'seconds'
