#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is a setup script for mysql_example.
It downloads a zip file of Illinois campaign contributions and loads them in to a MySQL
database named 'contributions'.

__Note:__ You will need to run this script first before execuing
[mysql_example.py](http://open-city.github.com/dedupe/doc/mysql_example.html).

Tables created:
* raw_table - raw import of entire CSV file
* donors - all distinct donors based on name and address
* recipients - all distinct campaign contribution recipients
* contributions - contribution amounts tied to donor and recipients tables
"""

import csv
import os
import urllib2
import zipfile
import time

import MySQLdb

start_time = time.time()

# Switch to our working directory
os.chdir('./examples/mysql_example/')

# Download Illinois-campaign-contributions.txt.zip from S3
contributions_zip_file = 'Illinois-campaign-contributions.txt.zip'
contributions_txt_file = 'Illinois-campaign-contributions.txt'

if not os.path.exists(contributions_zip_file) :
  print 'downloading', contributions_zip_file, '(~60mb) ...'
  u = urllib2.urlopen('https://s3.amazonaws.com/dedupe-data/Illinois-campaign-contributions.txt.zip')
  localFile = open(contributions_zip_file, 'w')
  localFile.write(u.read())
  localFile.close()

# Extract the zip file
if not os.path.exists(contributions_txt_file) :
  zip_file = zipfile.ZipFile(contributions_zip_file, 'r')
  print 'extracting %s' % contributions_zip_file
  zip_file_contents = zip_file.namelist()
  for f in zip_file_contents:
    if ('.txt' in f):
      zip_file.extract(f)
  zip_file.close()

# Create our database connection
# `local_infile` permission is required for inserting from a CSV file
conn = MySQLdb.connect(read_default_file = os.path.abspath('.') + '/mysql.cnf', 
                       local_infile = 1,
                       db='contributions')
c = conn.cursor()

print 'importing raw data from csv...'
c.execute("DROP TABLE IF EXISTS raw_table")
c.execute("DROP TABLE IF EXISTS donors")
c.execute("DROP TABLE IF EXISTS recipients")
c.execute("DROP TABLE IF EXISTS contributions")


c.execute("CREATE TABLE raw_table "
          "(reciept_id INT, last_name varchar(70), first_name varchar(35), "
          " address_1 varchar(35), address_2 varchar(36), city varchar(20), state varchar(15), "
          " zip varchar(11), report_type TEXT, date_recieved TEXT, "
          " loan_amount TEXT, amount TEXT, receipt_type TEXT, "
          " employer TEXT, occupation TEXT, vendor_last_name TEXT, "
          " vendor_first_name TEXT, vendor_address_1 TEXT, "
          " vendor_address_2 TEXT, vendor_city TEXT, vendor_state TEXT, "
          " vendor_zip TEXT, description TEXT, election_type TEXT, "
          " election_year TEXT, "
          " report_period_begin TEXT, report_period_end TEXT, "
          " committee_name TEXT, committee_id TEXT)")


conn.commit()

# Load in our data directly from the CSV file
c.execute("LOAD DATA LOCAL INFILE %s INTO TABLE raw_table "
          "FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\r\n' " 
          "IGNORE 1 LINES "
          "(reciept_id, last_name, first_name, "
          " address_1, address_2, city, state, "
          " zip, report_type, date_recieved, "
          " loan_amount, amount, receipt_type, "
          " employer, occupation, vendor_last_name, "
          " vendor_first_name, vendor_address_1, "
          " vendor_address_2, vendor_city, vendor_state, "
          " vendor_zip, description, election_type, "
          " election_year, "
          " report_period_begin, report_period_end, "
          " committee_name, committee_id, @dummy)",
          (os.path.abspath('.') + '/' + contributions_txt_file))

# Create our indexes. This is necessary for populating our contribution table
print 'creating raw_table indexes...'
c.execute("ALTER TABLE raw_table ADD PRIMARY KEY(reciept_id)")
# c.execute("CREATE INDEX raw_table_name_address_idx on raw_table (first_name, last_name, address_1, address_2, city, state, zip)")

conn.commit()

# Find all distinct donors based on name and address and insert into donors
print 'creating donors table...'
c.execute("CREATE TABLE donors "
          "(donor_id INTEGER PRIMARY KEY AUTO_INCREMENT, last_name varchar(70), first_name varchar(35), "
          " address_1 varchar(35), address_2 varchar(36), city varchar(20), state varchar(15), "
          " zip varchar(11))")
c.execute("INSERT INTO donors "
          "(first_name, last_name, address_1,"
          " address_2, city, state, zip) "
          "SELECT DISTINCT "
          "first_name, last_name, address_1, "
          "address_2, city, state, zip "
          "FROM raw_table")

print 'creating donors indexes...'
# c.execute("CREATE INDEX donors_name_address_idx on donors (first_name, last_name, address_1, address_2, city, state, zip)")

conn.commit()

# Find all distinct recipients based on committee and insert into recipients
print 'creating recipients table...'
c.execute("CREATE TABLE recipients "
          "(recipient_id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)")

c.execute("INSERT IGNORE INTO recipients "
          "SELECT DISTINCT committee_id, committee_name FROM raw_table")
conn.commit()

print 'creating contributions table'
c.execute('CREATE TABLE contributions '
          '(contribution_id INT, donor_id INT, recipient_id INT, '
          ' report_type TEXT, date_recieved TEXT, loan_amount TEXT, '
          ' amount TEXT, receipt_type TEXT, employer TEXT, '
          ' occupation TEXT, vendor_last_name TEXT, '
          ' vendor_first_name TEXT, vendor_address_1 TEXT, '
          ' vendor_address_2 TEXT, vendor_city TEXT, vendor_state TEXT, '
          ' vendor_zip TEXT, description TEXT, election_type TEXT, '
          ' election_year TEXT, report_period_begin TEXT, '
          ' report_period_end TEXT)')

# c.execute('INSERT INTO contributions '
# 'SELECT reciept_id, donors.donor_id, committee_id, '
# ' report_type, date_recieved, loan_amount, amount, '
# ' receipt_type, employer, occupation, vendor_last_name , '
# ' vendor_first_name, vendor_address_1, vendor_address_2, '
# ' vendor_city, vendor_state, vendor_zip, description, '
# ' election_type, election_year, report_period_begin, '
# ' report_period_end '
# 'FROM raw_table JOIN donors ON '
# 'CONCAT(donors.first_name, donors.last_name'
#        'donors.address_1, donors.address_2,'
#        'donors.city, donors.state, donors.zip) ='
# 'CONCAT(raw_table.first_name, raw_table.last_name'
#        'raw_table.address_1, raw_table.address_2,'
#        'raw_table.city, raw_table.state, raw_table.zip)')

# c.execute("ALTER TABLE contributions ADD PRIMARY KEY(contribution_id)")
# c.execute("CREATE INDEX donor_idx ON contributions (donor_id)")
# c.execute("CREATE INDEX recipient_idx ON contributions (recipient_id)")


# conn.commit()

c.close()
conn.close()
print 'done'

print 'ran in', time.time() - start_time, 'seconds'