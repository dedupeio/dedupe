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
before running this script. See the annotates source for 
[mysql_init_db](http://open-city.github.com/dedupe/doc/mysql_init_db.html)

For smaller datasets (<10,000), see our
[csv_example](http://open-city.github.com/dedupe/doc/csv_example.html)
"""
import os
import itertools
import time
import logging
import optparse
import locale

import MySQLdb
import MySQLdb.cursors

import dedupe

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. Added
# for convenience.  To enable verbose output, run `python
# examples/mysql_example/mysql_example.py -v`

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

# We'll be using variations on this following select statement to pull
# in campaign donor info.
#
# We are concatenating the first_name and last_name into a single name
# field and the address_1 and address_2 into a single address field. In
# this data, there is a lot of inconsitency of how these fields, so it
# is better to create these composite fields
#
# If a field is NULL, then we'll have MySQL return an empty string and
# if it's not null then we'll lower case the field
#
# Doing this preprocessing in MySQL is much faster than than in
# Python.
DONOR_SELECT = "SELECT donor_id, " \
               "IFNULL(LOWER(city), '') AS city, " \
               "LOWER(CONCAT_WS(' ', first_name, last_name)) AS name, " \
               "IFNULL(LOWER(zip),'') AS zip, " \
               "IFNULL(LOWER(state),'') AS state, " \
               "LOWER(CONCAT_WS(' ', address_1, address_2)) AS address " \
               "FROM donors"


def getSample(cur, sample_size, id_column, table):
    '''
    Returns a random sample of a given size of records pairs from a given
    MySQL table.
    '''

    cur.execute("SELECT MAX(%s) FROM %s" % (id_column, table))
    num_records = cur.fetchone().values()[0]

    # Dedupe expects the id column to contain unique, sequential
    # integers starting at 0 or 1
    random_pairs = dedupe.randomPairs(num_records,
                                      sample_size,
                                      zero_indexed=False)

    temp_d = {}

    cur.execute(DONOR_SELECT)
    for row in cur.fetchall() :
        temp_d[int(row[id_column])] = dedupe.core.frozendict(row)

    def random_pair_generator():
        for k1, k2 in random_pairs:
            yield (temp_d[k1], temp_d[k2])

    return tuple(pair for pair in random_pair_generator())

# Switch to our working directory and set up our settings and training
# file locations
os.chdir('./examples/mysql_example/')
settings_file = 'mysql_example_settings'
training_file = 'mysql_example_training.json'

start_time = time.time()

# You'll need to copy `examples/mysql_example/mysql.cnf_LOCAL` to
# `examples/mysql_example/mysql.cnf` and fill in your mysql database
# information in `examples/mysql_example/mysql.cnf`
con = MySQLdb.connect(db='contributions',
                      read_default_file = os.path.abspath('.') + '/mysql.cnf', 
                      cursorclass=MySQLdb.cursors.DictCursor)

c = con.cursor()

# ## Training

if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:

    # Select a large sample of duplicate pairs.  As the dataset grows,
    # duplicate pairs become relatively more rare so we have to take a
    # fairly large sample compared to `csv_example.py`
    print 'selecting random sample from donors table...'
    data_sample = getSample(c, 750000, 'donor_id', 'donors')


    # Define the fields dedupe will pay attention to
    #
    # The address, city, and zip fields are often missing, so we'll
    # tell dedupe that, and we'll learn a model that take that into
    # account
    fields = {'name': {'type': 'String'},
              'address': {'type': 'String', 'Has Missing' : True},
              'city': {'type': 'String', 'Has Missing' : True},
              'state': {'type': 'String'},
              'zip': {'type': 'String', 'Has Missing' : True},
              }

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # If we have training data saved from a previous run of dedupe,
    # look for it an load it in.
    #
    # __Note:__ if you want to train from
    # scratch, delete the training_file
    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_sample, training_file)

    # ## Active learning

    print 'starting active labeling...'
    # Starts the training loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    deduper.train(data_sample, dedupe.training.consoleLabel)

    # When finished, save our labeled, training pairs to disk
    deduper.writeTraining(training_file)

# ## Blocking

print 'blocking...'
# If we didn't read saved blocking rules from a settings file, then
# we will try to find good blocking rules now. In either case we'll
# initialize our blocker

# Notice our two arguments here
#
# `ppc` limits the Proportion of Pairs Covered that we allow a
# predicate to cover. If a predicate puts together a fraction of
# possible pairs greater than the ppc, that predicate will be removed
# from consideration. As the size of the data increases, the user
# will generally want to reduce ppc.
#
# `uncovered_dupes` is the number of true dupes pairs in our training
# data that we are willing to accept will never be put into any
# block. If true duplicates are never in the same block, we will never
# compare them, and may never declare them to be duplicates.
#
# However, requiring that we cover every single true dupe pair may
# mean that we have to use blocks that put together many, many
# distinct pairs that we'll have to expensively, compare as well.
blocker = deduper.blockingFunction(ppc=0.001, uncovered_dupes=5)

# Save our weights and predicates to disk.
deduper.writeSettings(settings_file)

# To run blocking on such a large set of data, we create a separate table
# that contains blocking keys and record ids
print 'creating blocking_map database'
c.execute("DROP TABLE IF EXISTS blocking_map")
c.execute("CREATE TABLE blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER)")


# If dedupe learned a TF-IDF blocking rule, we have to take a pass
# through the data and create TF-IDF canopies. This can take up to an
# hour
print 'creating inverted index'
c.execute(DONOR_SELECT)
full_data = ((row['donor_id'], row) for row in c.fetchall())
blocker.tfIdfBlocks(full_data)


# Now we are ready to write our blocking map table by creating a
# generator that yields unique `(block_key, donor_id)` tuples.
print 'writing blocking map'
def block_data() :
    c.execute(DONOR_SELECT)
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
        records = {}
        for row in c.fetchall() :
            records.update({row['donor_id'] : row})
        yield records

# Grab all the block keys with more than one record.
# These records will make up a block of records we will cluster.
blocking_key_sql = "SELECT block_key, COUNT(*) AS num_candidates " \
                   "FROM blocking_map " \
                   "GROUP BY block_key " \
                   "HAVING num_candidates > 1"

# Using a random sample of blocks we find our clustering threshold
# that maximizes the weighted average of our precision and recall
c.execute(blocking_key_sql + " ORDER BY RAND() LIMIT 1000")
sampled_block_keys = block_keys = (row['block_key'] for row in c.fetchall())
threshold = deduper.goodThreshold(candidates_gen(sampled_block_keys), .5)

# With our found threshold, and candidates generator, perform the
# clustering operation
c.execute(blocking_key_sql)
block_keys = (row['block_key'] for row in c.fetchall())

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(candidates_gen(block_keys),
                                            threshold)

# ## Writing out results

# We now have a sequence of tuples of donor ids that dedupe believes
# all refer to the same entity. We write this out onto an entity map
# table
c.execute("DROP TABLE IF EXISTS entity_map")

print 'creating entity_map database'
c.execute("CREATE TABLE entity_map "
          "(donor_id INTEGER, canon_id INTEGER, PRIMARY KEY(donor_id))")

for cluster in clustered_dupes :
    cluster_head = str(cluster.pop())
    c.execute('INSERT INTO entity_map VALUES (%s, %s)',
                (cluster_head, cluster_head))
    for key in cluster :
        c.execute('INSERT INTO entity_map VALUES (%s, %s)',
                    (str(key), cluster_head))

con.commit()

c.execute("CREATE INDEX head_index ON entity_map (canon_id)")
con.commit()

# Print out the number of duplicates found
print '# duplicate sets'
print len(clustered_dupes)

# ## Payoff

# With all this done, we can now begin to ask interesting questions
# of the data
#
# For example, let's see who the top 10 donors are.

locale.setlocale(locale.LC_ALL, '') # for pretty printing numbers

# Create a temporary table so each group and unmatched record has a unique id
c.execute("CREATE TEMPORARY TABLE e_map "
          "SELECT IFNULL(canon_id, donor_id) AS canon_id, donor_id "
          "FROM entity_map "
          "RIGHT JOIN donors USING(donor_id)")


c.execute("SELECT CONCAT_WS(' ', donors.first_name, donors.last_name) AS name, "
          "donation_totals.totals AS totals "
          "FROM donors INNER JOIN "
          "(SELECT canon_id, SUM(amount) AS totals "
          " FROM contributions INNER JOIN e_map "
          " USING (donor_id) "
          " GROUP BY (canon_id) "
          " ORDER BY totals "
          " DESC LIMIT 10) "
          "AS donation_totals "
          "WHERE donors.donor_id = donation_totals.canon_id")


print "Top Donors (deduped)"
for row in c.fetchall() :
    row['totals'] = locale.currency(row['totals'], grouping=True)
    print '%(totals)20s: %(name)s' % row

# Compare this to what we would have gotten if we hadn't done any
# deduplication
c.execute("SELECT CONCAT_WS(' ', donors.first_name, donors.last_name) as name, "
          "SUM(contributions.amount) AS totals "
          "FROM donors INNER JOIN contributions "
          "USING (donor_id) "
          "GROUP BY (donor_id) "
          "ORDER BY totals DESC "
          "LIMIT 10")

print "Top Donors (raw)"
for row in c.fetchall() :
    row['totals'] = locale.currency(row['totals'], grouping=True)
    print '%(totals)20s: %(name)s' % row



# Close our database connection
c.close()
con.close()

print 'ran in', time.time() - start_time, 'seconds'
