#!/usr/bin/python

import os
import sys
import csv
import re
import collections
import logging
import optparse

import re
import collections
import logging
import optparse


os.chdir('./examples/csv_example/')
input_file = 'csv_example_messy_input.csv'
output_file = 'custom_big.csv'

with open(input_file) as f, open(output_file, 'w') as w:
	reader = csv.DictReader(f)
	writer = csv.DictWriter(w, reader.fieldnames)
	writer.writeheader()
	for row in reader:
		writer.writerow(row)

        for i in range(1, 11):
                f.seek(0)
                reader.next()

	        for row in reader:
		        row['Id'] = int(row['Id']) + 3337*i
		        writer.writerow(row)
