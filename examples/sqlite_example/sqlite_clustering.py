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
import time
from collections import defaultdict
import itertools

os.chdir('./examples/sqlite_example/')
settings_file = 'sqlite_example_settings.json'

t0 = time.time()

con = sqlite3.connect("illinois_contributions.db")
con.row_factory = sqlite3.Row
con.execute("ATTACH DATABASE 'blocking_map.db' AS bm")
cur = con.cursor()


if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
  raise ValueError('Settings File Not Found')

# We grab all the block_keys with more than one record associated with
# it. These associated records will make up a block of records we will
# compare within.
blocking_key_sql = "SELECT key, COUNT(donor_id) AS num_candidates " \
                   "FROM bm.blocking_map GROUP BY key HAVING num_candidates > 1"

block_keys = (row['key'] for row in con.execute(blocking_key_sql))

# This grabs a block of records for comparison. We rely on the
# ordering of the donor_ids
donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors " \
               "INNER JOIN bm.blocking_map USING (donor_id) " \
               "WHERE key = ? ORDER BY donor_id"

# This generator yields blocks of data
def candidates_gen() :
    for i, block_key in enumerate(block_keys) :
        if i % 10000 == 0 :
          print i, "blocks"
          print time.time() - t0, "seconds"

        yield ((row['donor_id'], row) for row in con.execute(donor_select,
                                                             (block_key,)))

    
print 'clustering...'
clustered_dupes = deduper.duplicateClusters(candidates_gen())

# duplicateClusters gives us sequence of tuples of donor_ids that
# Dedupe believes all refer to the same entity. We write this out
# onto an entity map tbale
con.execute("DROP TABLE IF EXISTS entity_map")

print 'creating entity_map database'
con.execute("CREATE TABLE entity_map "
            "(donor_id TEXT, head_id TEXT, PRIMARY KEY(donor_id))")

for cluster in clustered_dupes :
    cluster_head = str(cluster.pop())
    cur.execute('INSERT INTO entity_map VALUES (?, ?)',
                (cluster_head, cluster_head))
    for key in cluster :
        cur.execute('INSERT INTO entity_map VALUES (?, ?)',
                    (str(key), cluster_head))

con.commit()

con.execute("CREATE INDEX head_index ON entity_map (head_id)")
con.commit()
        
print '# duplicate sets'
print len(clustered_dupes)

cur.close()
con.close()
print 'ran in', time.time() - t0, 'seconds'


