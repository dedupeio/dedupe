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
import itertools
import time
import logging
import optparse

import MySQLdb
import MySQLdb.cursors

import dedupe

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


def getSample(c, sample_size, id_column, table):
  """
  Returns a random sample of pairs of donors of a given size from a MySQL table.
  Depending on your database engine, you will need to come up with a similar function.

  id_column must contain unique, sequential itegers starting at 0 or 1
  """

  c.execute("SELECT MAX(%s) FROM %s" , (id_column, table))
  num_records = c.fetchone().values()[0]

  random_pairs = dedupe.randomPairs(num_records, sample_size, zero_indexed=False)

  temp_d = {}

  c.execute(donor_select) 
  for row in c.fetchall() :
    temp_d[int(row[id_column])] = dedupe.core.frozendict(row)

  def random_pair_generator():
    for record_id_1, record_id_2 in random_pairs:
      yield ((record_id_1, temp_d[record_id_1]),
             (record_id_2, temp_d[record_id_2]))
  
  return tuple(record_pairs for pair in random_pair_generator())


start_time = time.time()

# You'll need to copy `examples/mysql_example/mysql.cnf_LOCAL` to
# `examples/mysql_example/mysql.cnf` and put fill in your mysql
# database information examples/mysql_example/mysql.cnf
con = MySQLdb.connect(db='contributions',
                      read_default_file = os.path.abspath('.') + '/mysql.cnf', 
                      cursorclass=MySQLdb.cursors.DictCursor)

c = con.cursor()

# To run blocking on such a large set of data, we create a separate table
# that contains blocking keys and record ids
print 'creating blocking_map database'
c.execute("DROP TABLE IF EXISTS blocking_map")
c.execute("CREATE TABLE blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER)")


if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
    # As the dataset grows, duplicate pairs become more rare. 
    # We need positive examples to train dedupe, so we have to
    # signficantly increase the size of the sample compared to csv_example.py
    print 'selecting random sample from donors table...'
    data_sample = getSample(c, 750000, 'donor_id', 'donors')


    # Define the fields dedupe will pay attention to
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
blocker = deduper.blockingFunction(eta=0.001, epsilon=5)
deduper.writeSettings(settings_file)

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
# a generator that yields a (block_key, donor_id) tuples. Dedupe guarantees
# that these tuples will be unique
print 'writing blocking map'
def block_data() :
    c.execute(donor_select)
    full_data = ((row['donor_id'], row) for row in c.fetchall())
    for i, (donor_id, record) in enumerate(full_data) :
        if i % 10000 == 0 :
            print i, ',', time.time() - start_time, 'seconds'
        for key in blocker((donor_id, record)) :
            yield (key, donor_id)

b_data = block_data()

# MySQL, by default, has a hard limit on the size of a data object 
# that can be passed to it. To get around this, we chunk the blocked data
# in to groups of 10,000 blocks
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


# We create an index on the blocking_key so that the group by
# queries we will be making to cluster the data can  happen in a
# reasonable time
print 'creating blocking map index. this will probably take a while ...'
c.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (block_key)")
print 'created', time.time() - start_time, 'seconds'


# This grabs a block of records for comparison. We rely on the
# ordering of the donor_ids
donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors " \
               "INNER JOIN blocking_map USING (donor_id) " \
               "WHERE block_key = %s ORDER BY donor_id"


# This generator yields blocks of data
def candidates_gen(block_keys) :
    for i, block_key in enumerate(block_keys) :
        if i % 10000 == 0 :
          print i, "blocks"
          print time.time() - start_time, "seconds"

        c.execute(donor_select, (block_key,))
        yield ((row['donor_id'], row) for row in c.fetchall())


# We grab all the block_keys with more than one record associated with
# it. These associated records will make up a block of records we will
# compare within.
blocking_key_sql = "SELECT block_key, COUNT(*) AS num_candidates " \
                   "FROM blocking_map GROUP BY block_key HAVING num_candidates > 1"

# Using a random sample of blocks we find a good threshold
c.execute(blocking_key_sql + " ORDER BY RAND() LIMIT 1000")
sampled_block_keys = block_keys = (row['block_key'] for row in c.fetchall())
threshold = deduper.goodThreshold(candidates_gen(sampled_block_keys))


c.execute(blocking_key_sql)
block_keys = (row['block_key'] for row in c.fetchall())

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(candidates_gen(block_keys),
                                            threshold)

# duplicateClusters gives us sequence of tuples of donor_ids that
# Dedupe believes all refer to the same entity. We write this out
# onto an entity map tbale
c.execute("DROP TABLE IF EXISTS entity_map")

print 'creating entity_map database'
c.execute("CREATE TABLE entity_map "
          "(donor_id INTEGER, head_id INTEGER, PRIMARY KEY(donor_id))")

for cluster in clustered_dupes :
    cluster_head = str(cluster.pop())
    c.execute('INSERT INTO entity_map VALUES (%s, %s)',
                (cluster_head, cluster_head))
    for key in cluster :
        c.execute('INSERT INTO entity_map VALUES (%s, %s)',
                    (str(key), cluster_head))

con.commit()

c.execute("CREATE INDEX head_index ON entity_map (head_id)")
con.commit()
        
print '# duplicate sets'
print len(clustered_dupes)

c.close()
con.close()
print 'ran in', time.time() - start_time, 'seconds'


