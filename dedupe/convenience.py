#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for in memory deduplication
"""

import collections
import dedupe.core


def dataSample(data, sample_size):
    '''Randomly sample pairs of records from a data dictionary'''

    random_pairs = dedupe.core.randomPairs(len(data), sample_size)

    return tuple(((k_1, data[k_1]), (k_2, data[k_2])) for (k_1, k_2) in
                 random_pairs)


def blockData(data_d, blocker):

    blocks = collections.defaultdict(list)

    blocker.tfIdfBlocks(data_d.iteritems())

    for (record_id, record) in data_d.iteritems():
        for key in blocker((record_id, record)):
            blocks[key].append((record_id, record))

    return tuple(block for block in blocks.values())
        

        
