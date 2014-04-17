# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:04:10 2014

@author: nathanhoeft

__Note:__ You will need to run the pgsql_init_db.py script before executing this script.
"""

import dedupe
import os
import re
import collections
import time
import logging
import optparse

import psycopg2 as psy
import psycopg2.extras

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count', help='Increase verbosity (specify multiple times for more)')
(opts, args) = optp.parse_args()
log_level = logging.WARNING
if opts.verbose == 1:
    log_level = logging.INFO
elif opts.verbose >= 2:
    log_level = logging.DEBUG
logging.basicConfig(level=log_level)

settings_file = 'postgres_settings'
training_file = 'postgres_training.json'

start_time = time.time()

con = psy.connect(database='database', user = 'user', host='host', password='password')

con2 = psy.connect(database='database', user = 'user', host='host', password='password')

c = con.cursor(cursor_factory=psy.extras.RealDictCursor)

MAILING_SELECT = 'SELECT id,site_name, address, zip, phone FROM csv_messy_data'

def preProcess(column):
    column = dedupe.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column

print 'importing data ...'
c.execute(MAILING_SELECT)
data= c.fetchall()
data_d = {}
for row in data:
    clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
    row_id = int(row['id'])
    data_d[row_id] = dict(clean_row)

if os.path.exists(settings_file):
    print 'reading from', settings_file
    deduper = dedupe.StaticDedupe(settings_file)

else:
    fields = {
        'site_name': {'type': 'String'},
        'address': {'type': 'String'},
        'zip': {'type': 'String', 'Has Missing' : True},
        'phone': {'type': 'String', 'Has Missing' : True},
        }

    deduper = dedupe.Dedupe(fields)

    deduper.sample(data_d, 150000)

    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.readTraining(training_file)

    print 'starting active labeling...'

    dedupe.consoleLabel(deduper)

    deduper.train()
    
    deduper.writeTraining(training_file)

    deduper.writeSettings(settings_file)

print 'blocking...'

threshold = deduper.threshold(data_d, recall_weight=2)

print 'clustering...'
clustered_dupes = deduper.match(data_d, threshold)

print '# duplicate sets', len(clustered_dupes)

c2 = con2.cursor()
c2.execute('SELECT * FROM csv_messy_data')
data = c2.fetchall()

full_data = []

cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        for row in data:
            if record_id == int(row[0]):
                row = list(row)
                row.insert(0,cluster_id)
                row = tuple(row)
                full_data.append(row)

columns = "SELECT column_name FROM information_schema.columns WHERE table_name = 'csv_messy_data'"                
c2.execute(columns)
column_names = c2.fetchall()
column_names = [x[0] for x in column_names]
column_names.insert(0,'cluster_id')

c2.execute('DROP TABLE IF EXISTS deduped_table')
c2.execute('CREATE TABLE deduped_table (%s)'%','.join('%s varchar(200)' % name for name in column_names))
con2.commit()

num_cols = len(column_names)
mog = "(" + ("%s,"*(num_cols -1)) + "%s)"
args_str = ','.join(c2.mogrify(mog,x) for x in full_data)
values = "("+ ','.join(x for x in column_names) +")"
c2.execute("INSERT INTO deduped_table %s VALUES %s" % (values, args_str))
con2.commit()
con2.close()
con.close()

print 'ran in', time.time() - start_time, 'seconds'