import MySQLdb
import csv
from AsciiDammit import asciiDammit
import os
import urllib2
import zipfile

os.chdir('./examples/sqlite_example/')

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
                       db='contributions'
                       )
c = conn.cursor()

print 'importing raw data from csv...'
c.execute("DROP TABLE IF EXISTS raw_table")
c.execute("DROP TABLE IF EXISTS donors")
c.execute("DROP TABLE IF EXISTS recipients")
c.execute("DROP TABLE IF EXISTS contributions")


c.execute("CREATE TABLE raw_table "
          "(reciept_id INT, last_name TEXT, first_name TEXT, "
          " address_1 TEXT, address_2 TEXT, city TEXT, state TEXT, "
          " zip TEXT, report_type TEXT, date_recieved TEXT, "
          " loan_amount TEXT, amount TEXT, receipt_type TEXT, "
          " employer TEXT, occupation TEXT, vendor_last_name TEXT, "
          " vendor_first_name TEXT, vendor_address_1 TEXT, "
          " vendor_address_2 TEXT, vendor_city TEXT, vendor_state TEXT, "
          " vendor_zip TEXT, description TEXT, election_type TEXT, "
          " election_year TEXT, "
          " report_period_begin TEXT, report_period_end TEXT, "
          " committee_name TEXT, committee_id TEXT)")


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
          os.path.abspath('.') + '/' + contributions_txt_file)

c.execute("ALTER TABLE raw_table ADD PRIMARY KEY(reciept_id)")
conn.commit()

print 'creating donors table...'
c.execute("CREATE TABLE donors "
          "(donor_id INTEGER PRIMARY KEY AUTO_INCREMENT, first_name TEXT, "
          " last_name TEXT, address_1 TEXT, address_2 TEXT, "
          " city TEXT, state TEXT, zip TEXT)")
c.execute("INSERT INTO donors "
          "(first_name, last_name, address_1,"
          " address_2, city, state, zip) "
          "SELECT DISTINCT "
          "first_name, last_name, address_1, "
          "address_2, city, state, zip "
          "FROM raw_table")
conn.commit()

print 'creating recipients table...'
c.execute("CREATE TABLE recipients "
          "(recipient_id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)")

c.execute("INSERT IGNORE INTO recipients "
          "SELECT DISTINCT committee_id, committee_name FROM raw_table")
conn.commit()

## print 'creating contributions table'
## c.execute('CREATE TABLE contributions '
##           '(contribution_id INT, donor_id INT, recipient_id INT, '
##           ' report_type TEXT, date_recieved TEXT, loan_amount TEXT, '
##           ' amount TEXT, receipt_type TEXT, employer TEXT, '
##           ' occupation TEXT, vendor_last_name TEXT, '
##           ' vendor_first_name TEXT, vendor_address_1 TEXT, '
##           ' vendor_address_2 TEXT, vendor_city TEXT, vendor_state TEXT, '
##           ' vendor_zip TEXT, description TEXT, election_type TEXT, '
##           ' election_year TEXT, report_period_begin TEXT, '
##           ' report_period_end TEXT)')

## c.execute('INSERT INTO contributions '
## 'SELECT reciept_id, donors.donor_id, committee_id, '
## ' report_type, date_recieved, loan_amount, amount, '
## ' receipt_type, employer, occupation, vendor_last_name , '
## ' vendor_first_name, vendor_address_1, vendor_address_2, '
## ' vendor_city, vendor_state, vendor_zip, description, '
## ' election_type, election_year, report_period_begin, '
## ' report_period_end '
## 'FROM raw_table JOIN donors ON '
## 'donors.first_name = raw_table.first_name AND '
## 'donors.last_name = raw_table.last_name AND '
## 'donors.address_1 = raw_table.address_1 AND '
## 'donors.address_2 = raw_table.address_2 AND '
## 'donors.city = raw_table.city AND '
## 'donors.state = raw_table.state AND '
## 'donors.zip = raw_table.zip')

## c.execute("ALTER TABLE contributions ADD PRIMARY KEY(contribution_id)")
## c.execute("CREATE INDEX donor_idx ON contributions (donor_id)")
## c.execute("CREATE INDEX recipient_idx ON contributions (recipient_id)")


## conn.commit()

c.close()
conn.close()
print 'done'
