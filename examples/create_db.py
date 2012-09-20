import sqlite3
import csv
conn = sqlite3.connect("datasets/illinois_contributions.db")
c = conn.cursor()


c.execute("DROP TABLE IF EXISTS raw_table")

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

with open('datasets/markelReceipts.txt', 'rb') as f:
  contribution_reader = csv.reader(f, delimiter="\t")
  contribution_reader.next()
  i = 0
  for row in contribution_reader :
    c.execute('INSERT INTO raw_table VALUES '
              '(?, ?, ?, ?, ?, ?, ?, '
              ' ?, ?, ?, ?, ?, ?, ?, '
              ' ?, ?, ?, ?, ?, ?, ?, '
              ' ?, ?, ?, ?, ?, ?, ?, ?)',
              row[0:29])
    i += 1
    if i > 11 : break
conn.commit()

c.close()
          
          
