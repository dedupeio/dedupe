#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to deduplicate records when importing data into a database.

We start with the same CSV file utilized in csv_example.py and the output will be a 
postgresql table that is dynamically created from the header of the csv file. 

To understand what is being done with the data before the output to postgresql, please see csv_example.py.

For larger datasets, see our [mysql_example](http://open-city.github.com/dedupe/doc/mysql_example.html)
"""

import os
import csv
import re
import collections
import logging
import optparse
from numpy import nan
import psycopg2 as psy

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

input_file = 'csv_example_messy_input.csv'
settings_file = 'csv_example_learned_settings'
training_file = 'csv_example_training.json'

def sameOrNotComparator(field_1, field_2) :
    if field_1 and field_2 :
        if field_1 == field_2 :
            return 1
        else:
            return 0
    else :
        return nan

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of
    [AsciiDammit](https://github.com/tnajdek/ASCII--Dammit) and
    Regex. Things like casing, extra spaces, quotes and new lines can
    be ignored.
    """

    column = dedupe.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column


def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['Id'])
            data_d[row_id] = dict(clean_row)

    return data_d


print 'importing data ...'
data_d = readData(input_file)

if os.path.exists(settings_file):
    print 'reading from', settings_file
    deduper = dedupe.StaticDedupe(settings_file)

else:
    fields = {
        'Site name': {'type': 'String'},
        'Address': {'type': 'String'},
        'Zip': {'type': 'Custom', 
                'comparator' : sameOrNotComparator, 
                'Has Missing' : True},
        'Phone': {'type': 'String', 'Has Missing' : True},
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

# ## Writing Results

# Write our original data back out to a postgresql table with a new column called 
# 'cluster_id' which indicates which records refer to each other.

cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        cluster_membership[record_id] = cluster_id
            
results = []
with open(input_file) as f:
    reader = csv.reader(f)
    heading_row = reader.next()
    heading_row.insert(0,'cluster_id')
    heading_row = [re.sub(" ","_",x.lower()) for x in heading_row]
    num_cols = len(heading_row)
    mog = "(" + ("%s,"*(num_cols -1)) + "%s)"
    
    #Create connection to postgresql and open a cursor
    con= psy.connect(database = 'database', user = 'username', host = 'hostname', password = 'password')
    cur = con.cursor()
    #Query to get a list of reserved keywords to compare with header/column names
    cur.execute("SELECT word FROM pg_get_keywords() WHERE catdesc = 'reserved'")
    reserved = cur.fetchall()
    reserved = [x[0] for x in reserved]
    #Compare the column names in heading_row to reserved keywords to make sure there are no conflicts
    #and add "_" to any that do use reserved keywords
    heading_row = [(i+"_") if i in reserved else i for i in heading_row]
    
            
    cur.execute('CREATE TABLE csv_test (%s)'%','.join('%s varchar(200)' % name for name in heading_row))    
    con.commit()
    
    for row in reader:
        row_id = int(row[0])
        cluster_id = cluster_membership[row_id]
        row.insert(0, cluster_id)
        results.append(row)
        
    args_str = ','.join(cur.mogrify(mog,x) for x in results)
    header = "("+ ','.join(x for x in heading_row) +")"
    cur.execute("INSERT INTO csv_test %s VALUES %s" % (header, args_str))
    con.commit()
    con.close()
    