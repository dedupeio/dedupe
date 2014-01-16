#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

def consoleLabel(deduper):
    '''Command line interface for presenting and labeling training pairs by the user'''


    finished = False

    while not finished :
        uncertain_pairs = deduper.getUncertainPair()

        labels = {'distinct' : [], 'match' : []}

        for record_pair in uncertain_pairs:
            label = ''
            labeled = False

            for pair in record_pair:
                for field in deduper.data_model.comparison_fields:
                    line = "%s : %s\n" % (field, pair[field])
                    sys.stderr.write(line)
                sys.stderr.write('\n')

            sys.stderr.write('Do these records refer to the same thing?\n')

            valid_response = False
            while not valid_response:
                sys.stderr.write('(y)es / (n)o / (u)nsure / (f)inished\n')
                label = sys.stdin.readline().strip()
                if label in ['y', 'n', 'u', 'f']:
                    valid_response = True

            if label == 'y' :
                labels['match'].append(record_pair)
                labeled = True
            elif label == 'n' :
                labels['distinct'].append(record_pair)
                labeled = True
            elif label == 'f':
                sys.stderr.write('Finished labeling\n')
                finished = True
            elif label != 'u':
                sys.stderr.write('Nonvalid response\n')
                raise

        if labeled :
            deduper.markPairs(labels)
        

        
