#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for in memory deduplication
"""

import collections
import dedupe.core
try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict


def dataSampleConstrained(data_1, data_2, sample_size) :
     '''Randomly select pairs between two data dictionaries'''
     d_1 = dict((i, v) for i, v in enumerate(data_1.values()))
     d_2 = dict((i, v) for i, v in enumerate(data_2.values()))

     random_pairs = dedupe.core.randomPairsMatch(len(d_1),
                                                 len(d_2), 
                                                 sample_size)

     return tuple((d_1[int(k1)], 
                   d_2[int(k2)]) 
                  for k1, k2 in random_pairs)


def dataSample(data, sample_size):
    random_pairs = dedupe.core.randomPairs(len(data), 
                                           sample_size)

    return tuple((data.values()[int(k1)], 
                  data.values()[int(k2)]) 
                 for k1, k2 in random_pairs)


def blockData(data_d, blocker):

    blocks = OrderedDict({})

    blocker.tfIdfBlocks(data_d.iteritems())

    for (record_id, record) in data_d.iteritems():
        for key in blocker((record_id, record)):
            blocks.setdefault(key, {}).update({record_id : record})

    blocked_records = tuple(block for block in blocks.values())

    return blocked_records

def blockDataConstrained(data_1, data_2, blocker):

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


        

        
