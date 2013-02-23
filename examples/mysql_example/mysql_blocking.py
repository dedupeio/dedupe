#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is an example of working with very large data. There are about
700,000 unduplicated donors in this database of Illinois political
campaign contributions.

While we might be able to keep these donor records in memory, we
cannot possibly store all the comparison pairs we will make.

Because of performance issues that we are still working through, this
example is broken into two files, mysql_blocking.py which blocks the
data, mysql_clustering.py which clusters the blocked data.
"""
import os
import re
import MySQLdb
import MySQLdb.cursors
import dedupe
from collections import defaultdict
import itertools
import time
import os


os.chdir('./examples/mysql_example/')
settings_file = 'mysql_example_settings.json'
training_file = 'mysql_example_training.json'

# When we compare records, we don't care about differences in case.
# Lowering the case in SQL is much faster than in Python.
donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"


def getSample(c, size):
  """
  Returns a random sample of pairs of donors of a given size
  """

  c.execute("SELECT MAX(donor_id) FROM donors")
  dim = c.fetchone().values()[0]

  random_pairs = dedupe.randomPairs(dim, size, zero_indexed=False)

  temp_d = {}
  c.execute(donor_select) 
  for row in c.fetchall() :
    temp_d[int(row['donor_id'])] = dedupe.core.frozendict(row)

  return tuple((((record_id_1, temp_d[record_id_1]),
                 (record_id_2, temp_d[record_id_2]))
                for record_id_1, record_id_2
                in random_pairs))



t0 = time.time()

# You'll need to copy `examples/mysql_example/mysql.cnf_LOCAL` to
# `examples/mysql_example/mysql.cnf` and put fill in your mysql
# database information examples/mysql_example/mysql.cnf
con = MySQLdb.connect(db='contributions',
                       read_default_file = os.path.abspath('.') + '/mysql.cnf',
                       cursorclass=MySQLdb.cursors.DictCursor)

c = con.cursor()

print 'creating blocking_map database'
c.execute("DROP TABLE IF EXISTS blocking_map")
c.execute("CREATE TABLE blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER)")




if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
    # Unlike csv_example.py, we select from the database to get a random
    # sample for training. As the dataset grows, duplicate pairs become
    # more rare. We need positive examples to train dedupe, so we have to
    # signficantly increase the size of the sample
    print 'selecting random sample from donors table...'
    data_sample = getSample(c, 750000)


  
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
        deduper.train(data_sample, training_file)

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    # ... and then call training with our interactive function
    deduper.train(data_sample, dedupe.training.consoleLabel)
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
c.execute(donor_select)
full_data = ((row['donor_id'], row) for row in c.fetchall())
blocker.tfIdfBlocks(full_data)


# Finally, we are ready to block the data. We'll do this by creating
# a generator that yields a (block_key, donor_id) tuples. We guarantee
# that these tuples will be unique
print 'writing blocking map'
def block_data() :
    c.execute(donor_select)
    full_data = ((row['donor_id'], row) for row in c.fetchall())
    for i, (donor_id, record) in enumerate(full_data) :
        if i % 10000 == 0 :
            print i, ',', time.time() - t0, 'seconds'
        for key in blocker((donor_id, record)) :
            yield (key, donor_id)

b_data = block_data()

step = 10000
done = False
while not done :
  chunk = itertools.islice(b_data, step)
  # This takes the generator and writes into into our table
  records_written =  c.executemany("INSERT INTO blocking_map VALUES (%s, %s)",
                                   chunk)
  if records_written < step :
    done = True



  con.commit()


# Finally, we create an index on the blocking_key so that the group by
# queries we will be making in mysql_clustering can happen in a
# reasonable time
c.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (block_key)")
print 'created', time.time() - t0, 'seconds'

print 'ran in', time.time() - t0, 'seconds'

con.commit()
con.close()
