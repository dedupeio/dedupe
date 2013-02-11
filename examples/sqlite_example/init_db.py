import sqlite3
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


conn = sqlite3.connect("illinois_contributions.db")
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
          " committee_name TEXT, committee_id TEXT, "
          " PRIMARY KEY(reciept_id))")

conn.commit()

with open(contributions_txt_file, 'rb') as f:
  contribution_reader = csv.reader(f, delimiter="\t")
  contribution_reader.next()
  for row in contribution_reader :
    try:
      c.execute('INSERT INTO raw_table VALUES '
                '(?, ?, ?, ?, ?, ?, ?, ?, '
                ' ?, ?, ?, ?, ?, ?, ?, '
                ' ?, ?, ?, ?, ?, ?, ?, '
                ' ?, ?, ?, ?, ?, ?, ?)',
                row[0:29])
    except sqlite3.ProgrammingError: 
      try:
        c.execute('INSERT INTO raw_table VALUES '
                  '(?, ?, ?, ?, ?, ?, ?, ?, '
                  ' ?, ?, ?, ?, ?, ?, ?, '
                  ' ?, ?, ?, ?, ?, ?, ?, '
                  ' ?, ?, ?, ?, ?, ?, ?)',
                  [asciiDammit(field) for field in row[0:29]])

      except:
        print "failed to import row"
        print row
        raise

conn.commit()

print 'creating donors table...'
c.execute("CREATE TABLE donors "
          "(donor_id INTEGER PRIMARY KEY, first_name TEXT, "
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
          "(recipient_id INT, name TEXT, PRIMARY KEY(recipient_id))")

c.execute("INSERT INTO recipients "
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
          ' report_period_end TEXT, PRIMARY KEY(contribution_id))')

conn.commit()

c.close()
conn.close()
print 'done'
