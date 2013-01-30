#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
import time
from collections import defaultdict
import itertools


settings_file = 'sqlite_example_settings.json'

t0 = time.time()

con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = sqlite3.Row
con.execute("ATTACH DATABASE 'examples/sqlite_example/blocking_map.db' AS bm")
cur = con.cursor()


if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
  raise ValueError('Settings File Not Found')


block_keys = (row['key'] for row in con.execute('select key, count(donor_id) as num_candidates from bm.blocking_map group by key having num_candidates > 1 and num_candidates < 1000'))

donor_select = "SELECT donor_id, LOWER(city) AS city, " \
               "LOWER(first_name) AS first_name, " \
               "LOWER(last_name) AS last_name, " \
               "LOWER(zip) AS zip, LOWER(state) AS state, " \
               "LOWER(address_1) AS address_1, " \
               "LOWER(address_2) AS address_2 FROM donors"

# TODO: combine this with mergeBlocks
#@profile 
def candidates_gen() :
    for i, block_key in enumerate(block_keys) :
        if i % 10000 == 0 :
          print i, "blocks"
          print time.time() - t0, "seconds"

        yield ((row['donor_id'], row) for row in con.execute(donor_select + ' inner join bm.blocking_map using (donor_id) where key = ? order by donor_id', (block_key,)))

    
print 'clustering...'
clustered_dupes = deduper.duplicateClusters(candidates_gen())


con.execute("DROP TABLE IF EXISTS entity_map")

print 'creating entity_map database'
con.execute("CREATE TABLE entity_map "
            "(donor_id TEXT, head_id TEXT, PRIMARY KEY(donor_id))")
con.execute("CREATE INDEX head_index ON entity_map (head_id)")

for cluster in clustered_dupes :
    cluster_head = str(cluster.pop())
    cur.execute('INSERT INTO entity_map VALUES (?, ?)',
                (cluster_head, cluster_head))
    for key in cluster :
        cur.execute('INSERT INTO entity_map VALUES (?, ?)',
                    (str(key), cluster_head))

con.commit()

print '# duplicate sets'
print len(clustered_dupes)

cur.close()
con.close()
print 'ran in', time.time() - t0, 'seconds'

# select sum(amount), head_id from contributions inner join entity_map using (donor_id) group by head_id;
