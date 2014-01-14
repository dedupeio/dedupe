#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for in memory deduplication
"""

import collections
import dedupe.core
import sys

try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict


def dataSample(data, sample_size):
    d = dict((i, dedupe.core.frozendict(v)) 
             for i, v in enumerate(data.values()))

    random_pairs = dedupe.core.randomPairs(len(d), 
                                           sample_size)

    return tuple((d[int(k1)], 
                  d[int(k2)]) 
                 for k1, k2 in random_pairs)


def blockData(data_d, blocker):

    blocks = OrderedDict({})

    for field in blocker.tfidf_fields :
        blocker.tfIdfBlock(((record_id, record[field])
                            for record_id, record 
                            in data_d.iteritems()),
                           field)

    for block_key, record_id in blocker(data_d.iteritems()) :
        blocks.setdefault(block_key, {}).update({record_id : 
                                                 data_d[record_id]})

    blocked_records = tuple(block for block in blocks.values())

    return blocked_records

def dataSampleRecordLink(data_1, data_2, sample_size) :
     '''Randomly select pairs between two data dictionaries'''
     d_1 = dict((i, dedupe.core.frozendict(v)) 
                for i, v in enumerate(data_1.values()))
     d_2 = dict((i, dedupe.core.frozendict(v)) 
                for i, v in enumerate(data_2.values()))

     random_pairs = dedupe.core.randomPairsMatch(len(d_1),
                                                 len(d_2), 
                                                 sample_size)

     return tuple((d_1[int(k1)], 
                   d_2[int(k2)]) 
                  for k1, k2 in random_pairs)


def blockDataRecordLink(data_1, data_2, blocker):

    blocks = OrderedDict({})

    if not blocker.canopies : 
        blocker.tfIdfBlocks(data_1.items(), data_2.items())

    for (record_id, record) in data_1.iteritems():
        for key in blocker((record_id, record)):
            blocks.setdefault(key, ({},{}))[0].update({record_id : record})

    for (record_id, record) in data_2.iteritems():
        for key in blocker((record_id, record)):
            if key in blocks :
                blocks[key][1].update({record_id : record})

    for block in blocks.values () :
        yield block 


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
        

        
