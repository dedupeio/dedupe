#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import dedupe
import time
from collections import defaultdict
import itertools


class LowerRow(sqlite3.Row) :
  def __getitem__(self, key):
    return str(sqlite3.Row.__getitem__(self, key)).lower().strip()


settings_file = 'sqlite_example_settings.json'

t0 = time.time()

print 'selecting random sample from donors table...'
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

lower_con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
lower_con.row_factory = LowerRow
lower_con.execute("ATTACH DATABASE 'examples/sqlite_example/blocking_map.db' AS bm")


# TODO: combine this with mergeBlocks
#@profile 
def candidates_gen() :
    for i, block_key in enumerate(block_keys) :
        if i % 1000 == 0 :
          print i, "blocks"
          print time.time() - t0, "seconds"
        block = itertools.combinations(((row['donor_id'], row) for row in lower_con.execute('select * from donors inner join bm.blocking_map using (donor_id) where key = ? order by donor_id', (block_key,))), 2)
        for candidate in block :
          yield candidate

            


    
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
lower_con.close()
print 'ran in', time.time() - t0, 'seconds'

# select sum(amount), head_id from contributions inner join entity_map using (donor_id) group by head_id;
