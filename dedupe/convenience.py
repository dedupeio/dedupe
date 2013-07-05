#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for in memory deduplication
"""

import collections
import dedupe.core

    

def dataSample(data, sample_size):
    '''Randomly sample pairs of records from a data dictionary'''

    data_list = data.values()
    random_pairs = dedupe.core.randomPairs(len(data_list), sample_size)

    return tuple((data_list[k1], data_list[k2]) for k1, k2 in random_pairs)


def blockData(data_d, blocker):

    blocks = dedupe.core.OrderedDict({})
    record_blocks = dedupe.core.OrderedDict({})
    key_blocks = dedupe.core.OrderedDict({})

    blocker.tfIdfBlocks(data_d.iteritems())

    for (record_id, record) in data_d.iteritems():
        for key in blocker((record_id, record)):
            blocks.setdefault(key, {}).update({record_id : record})

    blocked_records = tuple(block for block in blocks.values())

    return blocked_records
        

        
