 #!/usr/bin/python
 # -*- coding: utf-8 -*-
"""
This is a setup script for mysql_example.  It downloads a zip file of
Illinois campaign contributions and loads them in t aMySQL database
named 'contributions'.
 
__Note:__ You will need to run this script first before execuing
[mysql_example.py](http://open-city.github.com/dedupe/doc/mysql_example.html).
 
Tables created:
* raw_table - raw import of entire CSV file
* donors - all distinct donors based on name and address
* recipients - all distinct campaign contribution recipients
* contributions - contribution amounts tied to donor and recipients tables
"""

import os
import urllib2
import zipfile
import warnings

import MySQLdb

warnings.filterwarnings('ignore', category=MySQLdb.Warning)

os.chdir('./examples/mysql_example/')

contributions_zip_file = 'Illinois-campaign-contributions.txt.zip'
contributions_txt_file = 'Illinois-campaign-contributions.txt'

if not os.path.exists(contributions_zip_file) :
    print 'downloading', contributions_zip_file, '(~60mb) ...'
    u = urllib2.urlopen('https://s3.amazonaws.com/dedupe-data/Illinois-campaign-contributions.txt.zip')
    localFile = open(contributions_zip_file, 'w')
    localFile.write(u.read())
    localFile.close()

if not os.path.exists(contributions_txt_file) :
    zip_file = zipfile.ZipFile(contributions_zip_file, 'r')
    print 'extracting %s' % contributions_zip_file
    zip_file_contents = zip_file.namelist()
    for f in zip_file_contents:
        if ('.txt' in f):
            zip_file.extract(f)
    zip_file.close()

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
          "(reciept_id INT, last_name VARCHAR(70), first_name VARCHAR(35), "
          " address_1 VARCHAR(35), address_2 VARCHAR(36), city VARCHAR(20), "
          " state VARCHAR(15), zip VARCHAR(11), report_type VARCHAR(24), "
          " date_recieved VARCHAR(10), loan_amount VARCHAR(12), "
          " amount VARCHAR(23), receipt_type VARCHAR(23), "
          " employer VARCHAR(70), occupation VARCHAR(40), "
          " vendor_last_name VARCHAR(70), vendor_first_name VARCHAR(20), "
          " vendor_address_1 VARCHAR(35), vendor_address_2 VARCHAR(31), "
          " vendor_city VARCHAR(20), vendor_state VARCHAR(10), "
          " vendor_zip VARCHAR(10), description VARCHAR(90), "
          " election_type VARCHAR(10), election_year VARCHAR(10), "
          " report_period_begin VARCHAR(10), report_period_end VARCHAR(33), "
          " committee_name VARCHAR(70), committee_id VARCHAR(37)) "
          "CHARACTER SET utf8 COLLATE utf8_unicode_ci")


conn.commit()

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
conn.commit()

print 'creating donors table...'
c.execute("CREATE TABLE donors "
          "(donor_id INTEGER PRIMARY KEY AUTO_INCREMENT, "
          " last_name VARCHAR(70), first_name VARCHAR(35), "
          " address_1 VARCHAR(35), address_2 VARCHAR(36), "
          " city VARCHAR(20), state VARCHAR(15), "
          " zip VARCHAR(11)) "
          "CHARACTER SET utf8 COLLATE utf8_unicode_ci")
c.execute("INSERT INTO donors "
          "(first_name, last_name, address_1,"
          " address_2, city, state, zip) "
          "SELECT DISTINCT "
          "TRIM(first_name), TRIM(last_name), TRIM(address_1), "
          "TRIM(address_2), TRIM(city), TRIM(state), TRIM(zip) "
          "FROM raw_table")
conn.commit()


print 'creating indexes on donors table'
c.execute("CREATE INDEX donors_donor_info ON donors "
          "(last_name, first_name, address_1, address_2, city, "
          " state, zip)")
conn.commit()



print 'creating recipients table...'
c.execute("CREATE TABLE recipients "
          "(recipient_id INTEGER PRIMARY KEY AUTO_INCREMENT, name VARCHAR(70)) "
          "CHARACTER SET utf8 COLLATE utf8_unicode_ci")

c.execute("INSERT IGNORE INTO recipients "
          "SELECT DISTINCT committee_id, committee_name FROM raw_table")
conn.commit()

print 'creating contributions table'
c.execute("CREATE TABLE contributions "
          "(contribution_id INT, donor_id INT, recipient_id INT, "
          " report_type VARCHAR(24), date_recieved DATE, "
          " loan_amount VARCHAR(12), amount VARCHAR(23), "
          " receipt_type VARCHAR(23), employer VARCHAR(70), "
          " occupation VARCHAR(40), vendor_last_name VARCHAR(70), "
          " vendor_first_name VARCHAR(20), "
          " vendor_address_1 VARCHAR(35), vendor_address_2 VARCHAR(31), "
          " vendor_city VARCHAR(20), vendor_state VARCHAR(10), "
          " vendor_zip VARCHAR(10), description VARCHAR(90), "
          " election_type VARCHAR(10), election_year VARCHAR(10), "
          " report_period_begin DATE, report_period_end DATE) "
          "CHARACTER SET utf8 COLLATE utf8_unicode_ci")


c.execute("INSERT INTO contributions "
          "SELECT reciept_id, donors.donor_id, committee_id, "
          " report_type, STR_TO_DATE(date_recieved, '%m/%d/%Y'), "
          " loan_amount, amount, "
          " receipt_type, employer, occupation, vendor_last_name , "
          " vendor_first_name, vendor_address_1, vendor_address_2, "
          " vendor_city, vendor_state, vendor_zip, description, "
          " election_type, election_year, "
          " STR_TO_DATE(report_period_begin, '%m/%d/%Y'), "
          " STR_TO_DATE(report_period_end, '%m/%d/%Y') "
          "FROM raw_table JOIN donors ON "
          "donors.first_name = TRIM(raw_table.first_name) AND "
          "donors.last_name = TRIM(raw_table.last_name) AND "
          "donors.address_1 = TRIM(raw_table.address_1) AND "
          "donors.address_2 = TRIM(raw_table.address_2) AND "
          "donors.city = TRIM(raw_table.city) AND "
          "donors.state = TRIM(raw_table.state) AND "
          "donors.zip = TRIM(raw_table.zip)")
conn.commit()

print 'creating indexes on contributions'
c.execute("ALTER TABLE contributions ADD PRIMARY KEY(contribution_id)")
c.execute("CREATE INDEX donor_idx ON contributions (donor_id)")
c.execute("CREATE INDEX recipient_idx ON contributions (recipient_id)")


conn.commit()

print 'nullifying empty strings in donors'
c.execute("UPDATE donors "
          "SET "
          "first_name = CASE first_name WHEN '' THEN NULL ELSE first_name END, "
          "last_name = CASE last_name WHEN '' THEN NULL ELSE last_name END, "
          "address_1 = CASE address_1 WHEN '' THEN NULL ELSE address_1 END, "
          "address_2 = CASE address_2 WHEN '' THEN NULL ELSE address_2 END, "
          "city = CASE city WHEN '' THEN NULL ELSE city END, "
          "state = CASE state WHEN '' THEN NULL ELSE state END, "
          "zip = CASE zip WHEN '' THEN NULL ELSE zip END")


conn.commit()


c.close()
conn.close()
print 'done'
