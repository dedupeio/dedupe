#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for in memory deduplication
"""

import collections
import dedupe.core

    

def dataSample(data, sample_size, constrained_matching=False):
    '''Randomly sample pairs of records from a data dictionary'''

    data_list = data.values()

    if constrained_matching:
        data_list_A = []
        data_list_B = []

        for record in data_list:
            if record['dataset'] == 0:
                data_list_A.append(record)
            else:
                data_list_B.append(record)

        random_pairs = dedupe.core.randomPairsMatch(len(data_list_A),len(data_list_B), sample_size)

        return tuple((data_list_A[k1], data_list_B[k2]) for k1, k2 in random_pairs)
    else:
        random_pairs = dedupe.core.randomPairs(len(data_list), sample_size)

        return tuple((data_list[k1], data_list[k2]) for k1, k2 in random_pairs)


def blockData(data_d, blocker, constrained_matching=False):

    blocks = dedupe.core.OrderedDict({})
    record_blocks = dedupe.core.OrderedDict({})
    key_blocks = dedupe.core.OrderedDict({})

    blocker.tfIdfBlocks(data_d.iteritems(), constrained_matching)

    for (record_id, record) in data_d.iteritems():
        for key in blocker((record_id, record)):
            blocks.setdefault(key, {}).update({record_id : record})

    blocked_records = tuple(block for block in blocks.values())

    return blocked_records
        

        
