#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is a setup script for pgsql_example. It loads the data in a csv file into a 
a postgresql table named csv_messy_data.
 
__Note:__ You will need to run this script first before executing [pgsql_example.py]
 
Tables created:
* csv_messy_data - raw import of entire CSV file
"""

import csv
import re

import psycopg2

input_file = 'csv_example_messy_input.csv'

con = psycopg2.connect(database='fish_bowl_spirits', user = 'mship', host='mshipdbinstance.cc5me1g5ormb.us-west-2.rds.amazonaws.com', password='Fl0th!nkery')
    
cur = con.cursor()

print 'importing raw data from csv...'

results = []
with open(input_file) as f:
    reader = csv.reader(f)
    heading_row = reader.next()
    heading_row = [re.sub(" ","_",x.lower()) for x in heading_row]
    num_cols = len(heading_row)
    mog = "(" + ("%s,"*(num_cols -1)) + "%s)"
    
    #Query to get a list of reserved keywords to compare with header/column names
    cur.execute("select word from pg_get_keywords() where catdesc = 'reserved'")
    reserved = cur.fetchall()
    reserved = [x[0] for x in reserved]
    #Compare the column names in heading_row to reserved keywords to make sure there are no conflicts
    #and add "_" to any that do use reserved keywords
    heading_row = [(i+"_") if i in reserved else i for i in heading_row]
    
            
    cur.execute('create table csv_messy_data (%s)'%','.join('%s varchar(200)' % name for name in heading_row))    
    con.commit()
    
    for row in reader:
        results.append(row)
        
    args_str = ','.join(cur.mogrify(mog,x) for x in results)
    header = "("+ ','.join(x for x in heading_row) +")"
    cur.execute("insert into csv_messy_data %s values %s" % (header, args_str))
    con.commit()
    con.close()

cur.close()
con.close()
print 'done'
