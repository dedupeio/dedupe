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
import pickle
import multiprocessing

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
MYSQL_CNF = os.path.abspath('.') + '/mysql.cnf'

settings_file = 'mysql_example_settings'
training_file = 'mysql_example_training.json'

start_time = time.time()

# You'll need to copy `examples/mysql_example/mysql.cnf_LOCAL` to
# `examples/mysql_example/mysql.cnf` and fill in your mysql database
# information in `examples/mysql_example/mysql.cnf`

# We use Server Side cursors (SSDictCursor and SSCursor) to [avoid
# having to have enormous result sets in memory](http://stackoverflow.com/questions/1808150/how-to-efficiently-use-mysqldb-sscursor).
con = MySQLdb.connect(db='contributions',
                      charset='ascii',
                      read_default_file = MYSQL_CNF, 
                      cursorclass=MySQLdb.cursors.SSDictCursor)
c = con.cursor()
c.execute("SET net_write_timeout = 3600")

con2 = MySQLdb.connect(db='contributions',
                       charset='ascii',
                       read_default_file = MYSQL_CNF, 
                       cursorclass=MySQLdb.cursors.SSCursor)
c2 = con2.cursor()
c2.execute("SET net_write_timeout = 3600")


# We'll be using variations on this following select statement to pull
# in campaign donor info.
#
# We did a fair amount of preprocessing of the fields in
# `mysql_init_db.py`

DONOR_SELECT = "SELECT donor_id, city, name, zip, state, address, " \
               "occupation, employer, person from processed_donors"


def getSample(cur, sample_size, id_column, table):
    '''
    Returns a random sample of a given size of records pairs from a given
    MySQL table.
    '''

    cur.execute("SELECT MAX(%s) FROM %s" % (id_column, table))
    num_records = cur.fetchone().values()[0]

    cur.fetchall()

    random_pairs = dedupe.randomPairs(num_records,
                                      sample_size) 
    random_pairs += 1

    temp_d = {}

    cur.execute(DONOR_SELECT)
    for row in cur :
        temp_d[int(row[id_column])] = dedupe.core.frozendict(row)

    def random_pair_generator():
        for k1, k2 in random_pairs:
            yield (temp_d[k1], temp_d[k2])

    return tuple(pair for pair in random_pair_generator())


# ## Training

if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.StaticDedupe(settings_file, num_processes=2)
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
              'person' : {'type' : 'Categorical', 
                          'Categories' : [0, 1]},
              'person-address' : {'type' : 'Interaction',
                                  'Interaction Fields' : ['person', 'address']},
              'name-address' : {'type' : 'Interaction', 
                                'Interaction Fields' : ['name', 'address']}
              }

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields, data_sample, num_processes=4)

    # If we have training data saved from a previous run of dedupe,
    # look for it an load it in.
    #
    # __Note:__ if you want to train from
    # scratch, delete the training_file
    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.readTraining(training_file)

    # ## Active learning

    print 'starting active labeling...'
    # Starts the training loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    dedupe.convenience.consoleLabel(deduper)

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
    deduper.train(ppc=0.001, uncovered_dupes=5)

    # When finished, save our labeled, training pairs to disk
    deduper.writeTraining(training_file)
    deduper.writeSettings(settings_file)

# ## Blocking

print 'blocking...'

# To run blocking on such a large set of data, we create a separate table
# that contains blocking keys and record ids
print 'creating blocking_map database'
c.execute("DROP TABLE IF EXISTS blocking_map")
c.execute("DROP TABLE IF EXISTS sorted_blocking_map")
c.execute("CREATE TABLE blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER) "
          "CHARACTER SET utf8 COLLATE utf8_unicode_ci")


# If dedupe learned a TF-IDF blocking rule, we have to take a pass
# through the data and create TF-IDF canopies. This can take up to an
# hour
print 'creating inverted index'


for field in deduper.blocker.tfidf_fields :
    c2.execute("SELECT donor_id, %s FROM processed_donors" % field)
    field_data = (row for row in c2)
    deduper.blocker.tfIdfBlock(field_data, field)

# Now we are ready to write our blocking map table by creating a
# generator that yields unique `(block_key, donor_id)` tuples.
print 'writing blocking map'

c.execute(DONOR_SELECT)
full_data = ((row['donor_id'], row) for row in c)
b_data = deduper.blocker(full_data)

# MySQL has a hard limit on the size of a data object that can be
# passed to it.  To get around this, we chunk the blocked data in
# to groups of 30,000 blocks
step_size = 30000

# We will also speed up the writing by of blocking map by using 
# parallel database writers
def dbWriter(sql, rows) :
    conn = MySQLdb.connect(db='contributions',
                           charset='ascii',
                           read_default_file = MYSQL_CNF) 

    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    cursor.close()
    conn.commit()
    conn.close()

pool = dedupe.Pool(processes=2)

done = False

while not done :
    chunks = (list(itertools.islice(b_data, step)) for step in [step_size]*100)

    results =[pool.apply_async(dbWriter,
                               ("INSERT INTO blocking_map VALUES (%s, %s)", 
                                chunk))
              for chunk in chunks]

    for r in results :
        r.wait()

    if len(chunk) < step_size :
        done = True

pool.close()

# Remove blocks that contain only one record, sort by block key and
# donor, key and index blocking map.
#
# These steps, particularly the sorting will let us quickly create
# blocks of data for comparison
print 'creating blocking map index. this will probably take a while ...'

c.execute("CREATE INDEX blocking_map_key_idx ON blocking_map (block_key)")
print 'created', time.time() - start_time, 'seconds'

print "calculating plural_key"
c.execute("CREATE TABLE plural_key "
          "(SELECT block_key FROM blocking_map "
          " GROUP BY block_key HAVING COUNT(*) > 1)")

c.execute("CREATE INDEX block_key_idx ON plural_key (block_key)")

print "filtering singleton blocks"
c.execute("CREATE TABLE plural_block "
          "(SELECT block_key, donor_id FROM blocking_map INNER JOIN plural_key "
          " USING (block_key))")

c.execute("CREATE INDEX sorting_key "
          "ON plural_block (block_key, donor_id)")

print "creating sorted_blocking_map"
c.execute("CREATE TABLE sorted_blocking_map "
          "(block_key VARCHAR(200), donor_id INTEGER, "
          " id INT(8) UNSIGNED AUTO_INCREMENT, "
          " PRIMARY KEY (id)) "
          "(SELECT block_key, donor_id FROM plural_block "
          " ORDER BY block_key, donor_id)")

c.execute("CREATE INDEX donor_idx ON sorted_blocking_map (donor_id)")

c.execute("DROP TABLE plural_key")
c.execute("DROP TABLE plural_block")

con.commit()


## Clustering

def candidates_gen(result_set) :

    block_key = None
    records = {}
    i = 0
    for row in result_set :
        if row['block_key'] != block_key :
            if records :
                yield records

            block_key = row['block_key']
            records = {}
            i += 1

            if i % 10000 == 0 :
                print i, "blocks"
                print time.time() - start_time, "seconds"

            
        records.update({row['donor_id'] : row})

    if records :
        yield records

c.execute("SELECT donor_id, city, name, zip, state, address, "
          "occupation, employer, person, block_key from processed_donors "
          "INNER JOIN sorted_blocking_map using (donor_id) "
          "ORDER BY sorted_blocking_map.id")

print 'clustering...'
clustered_dupes = deduper.matchBlocks(candidates_gen(c),
                                      threshold=0.5)

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
