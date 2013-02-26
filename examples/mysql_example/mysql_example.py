#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is an example of working with very large data. There are about
700,000 unduplicated donors in this database of Illinois political
campaign contributions.

With such a large set of input data, we cannot store all the comparisons 
we need to make in memory. Instead, we will read the pairs on demand
from the MySQL database.

__Note:__ You will need to run `python examples/mysql_example/mysql_init_db.py` 
before running this script. See the annotated source for 
[mysql_init_db.py](http://open-city.github.com/dedupe/doc/mysql_init_db.html)

For smaller datasets (<10,000), see our [csv_example](http://open-city.github.com/dedupe/doc/csv_example.html)
"""
import os
import itertools
import time
import logging
import optparse

import MySQLdb
import MySQLdb.cursors

import dedupe

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. Added for convenience.
# To enable verbose logging, run `python examples/mysql_example/mysql_example.py -v`

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

# Create a select function that pulls in our campaign donor info.
# When we compare records, we don't care about differences in casing,
# so we lower the case (doing this in SQL is much faster than in Python).
DONOR_SELECT = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"


def getSample(c, sample_size, id_column, table):
  '''
  Returns a random sample of a given size of records pairs from a given
  MySQL table.
  '''

  c.execute("SELECT MAX(%s) FROM %s" , (id_column, table))
  num_records = c.fetchone().values()[0]

  # dedupe expects the id column to contain unique, sequential itegers starting at 0 or 1
  random_pairs = dedupe.randomPairs(num_records, sample_size, zero_indexed=False)

  temp_d = {}

  c.execute(DONOR_SELECT) 
  for row in c.fetchall() :
    temp_d[int(row[id_column])] = dedupe.core.frozendict(row)

  def random_pair_generator():
    for record_id_1, record_id_2 in random_pairs:
      yield ((record_id_1, temp_d[record_id_1]),
             (record_id_2, temp_d[record_id_2]))
  
  return tuple(record_pairs for pair in random_pair_generator())

# Switch to our working directory and set up our settings and training file locations
os.chdir('./examples/mysql_example/')
settings_file = 'mysql_example_settings.json'
training_file = 'mysql_example_training.json'

start_time = time.time()

# ## Create a database connection

# You'll need to copy `examples/mysql_example/mysql.cnf_LOCAL` to
# `examples/mysql_example/mysql.cnf` and fill in your mysql
# database information in `examples/mysql_example/mysql.cnf`
con = MySQLdb.connect(db='contributions',
                      read_default_file = os.path.abspath('.') + '/mysql.cnf', 
                      cursorclass=MySQLdb.cursors.DictCursor)

c = con.cursor()

# ## Training

if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:

    # Select a large sample of duplicate pairs.
    # As the dataset grows, duplicate pairs become relatively more
    # rare so we have to take a fairly large sample compared to
    # `csv_example.py`
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

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # If we have training data saved from a previous run of dedupe,
    # look for it an load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_sample, training_file)

    # ## Active learning

    print 'starting active labeling...'
    print 'finding uncertain pairs...'
    
    # Starts the trainin loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    deduper.train(data_sample, dedupe.training.consoleLabel)

    # When finished, save our training away to disk
    deduper.writeTraining(training_file)

# ## Blocking

print 'blocking...'
# Initialize our blocker, which determines our field weights and blocking 
# predicates based on our training data
blocker = deduper.blockingFunction(eta=0.001, epsilon=5)

# Save our weights and predicates to disk.
# If the settings file exists, we will skip all the training and learning
deduper.writeSettings(settings_file)

# Iterate through all our input data and create a blocking map.
# To run blocking on such a large set of data, we create a separate table
# that contains blocking keys and record ids
print 'creating blocking_map database'
c.execute("DROP TABLE IF EXISTS blocking_map")
c.execute("CREATE TABLE blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER)")


# We are using [TF/IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) as a 
# potential predicate. 
# If dedupe learned a TF-IDF blocking rule, we create go through the extra
# step of creating TF-IDF canopies.
print 'creating inverted index'
c.execute(DONOR_SELECT + " LIMIT 10000")
full_data = ((row['donor_id'], row) for row in c.fetchall())
blocker.tfIdfBlocks(full_data)


# Next, we write our blocking map table by creating a generator that 
# yields unique `(block_key, donor_id)` tuples.
print 'writing blocking map'
def block_data() :
    c.execute(DONOR_SELECT + " LIMIT 10000")
    full_data = ((row['donor_id'], row) for row in c.fetchall())
    for i, (donor_id, record) in enumerate(full_data) :
        if i % 10000 == 0 :
            print i, ',', time.time() - start_time, 'seconds'
        for key in blocker((donor_id, record)) :
            yield (key, donor_id)

b_data = block_data()

# MySQL has a hard limit on the size of a data object that can be passed to it. 
# To get around this, we chunk the blocked data in to groups of 10,000 blocks
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


# Create an index on the blocking key for faster clustering
print 'creating blocking map index. this will probably take a while ...'
c.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (block_key)")
print 'created', time.time() - start_time, 'seconds'

# ## Clustering

# Grabs a block of records for comparison.
block_select = (DONOR_SELECT + 
                " INNER JOIN blocking_map USING (donor_id) " 
                "WHERE block_key = %s ORDER BY donor_id")


# Generator function that yields records based on blocking map keys
def candidates_gen(block_keys) :
    for i, block_key in enumerate(block_keys) :
        if i % 10000 == 0 :
          print i, "blocks"
          print time.time() - start_time, "seconds"

        c.execute(block_select, (block_key,))
        yield ((row['donor_id'], row) for row in c.fetchall())

# Grab all the block keys with more than one record.
# These records will make up a block of records we will cluster.
blocking_key_sql = "SELECT block_key, COUNT(*) AS num_candidates " \
                   "FROM blocking_map GROUP BY block_key HAVING num_candidates > 1"

# Using a random sample of blocks we find our clustering threshold that maximizes
# the weighted average of our precision and recall
c.execute(blocking_key_sql + " ORDER BY RAND() LIMIT 1000")
sampled_block_keys = block_keys = (row['block_key'] for row in c.fetchall())
threshold = deduper.goodThreshold(candidates_gen(sampled_block_keys))

# With our found threshold, and candidates generator, perform the clustering operation
c.execute(blocking_key_sql)
block_keys = (row['block_key'] for row in c.fetchall())

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(candidates_gen(block_keys),
                                            threshold)

# ## Writing out results

# We now have a sequence of tuples of donor ids that dedupe believes all 
# refer to the same entity. We write this out onto an entity map table
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

# Print out the number of duplicates found
print '# duplicate sets'
print len(clustered_dupes)

# Close our database connection
c.close()
con.close()

print 'ran in', time.time() - start_time, 'seconds'