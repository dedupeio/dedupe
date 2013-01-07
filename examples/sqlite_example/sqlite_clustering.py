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

print 'selecting random sample from donors table...'
con = sqlite3.connect("examples/sqlite_example/illinois_contributions.db")
con.row_factory = sqlite3.Row
cur = con.cursor()


if os.path.exists(settings_file):
    print 'reading from ', settings_file
    deduper = dedupe.Dedupe(settings_file)
else:
  raise ValueError('Settings File Not Found')



cur.execute('select * from donors join '
 '(select key, donor_id from blocking_map '
 'join (select key, count(donor_id) num_candidates from blocking_map '
 'group by key having num_candidates > 1) '
 'as bucket using (key)) as candidates using (donor_id)')

block_keys = (row['key'] for row in con.execute('select key, count(donor_id) as num_candidates from blocking_map group by key having num_candidates > 1'))


def candidates_gen() :
    candidate_set = set([])
    for block_key in block_keys :
        block = set(itertools.combinations(((row['donor_id'], row) for row in con.execute('select * from donors inner join blocking_map using (donor_id) where key = ? order by donor_id', (block_key,))), 2))
        new = block - candidate_set
        candidate_set |= new
        for candidate_pair in new :
            yield candidate_pair

# for i, pair in enumerate(candidates_gen()):
#      print pair

    
print 'clustering...'
clustered_dupes = deduper.duplicateClustersII(candidates_gen())

print '# duplicate sets'
print len(clustered_dupes)

cur.close()
con.close()
print 'ran in', time.time() - t0, 'seconds'
