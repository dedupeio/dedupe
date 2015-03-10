#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from builtins import input

import collections
import itertools
import random
from dedupe.core import randomPairs
from canonicalize.centroid import getCanonicalRep

def consoleLabel(deduper): # pragma : no cover
    '''
    Command line interface for presenting and labeling training pairs
    by the user
    
    Argument :
    A deduper object
    '''

    finished = False

    while not finished :
        uncertain_pairs = deduper.uncertainPairs()

        labels = {'distinct' : [], 'match' : []}

        for record_pair in uncertain_pairs:
            label = ''
            labeled = False

            for pair in record_pair:
                for field in set(field[0] for field 
                                 in deduper.data_model.field_comparators) :
                    line = "%s : %s" % (field, pair[field])
                    print(line)
                print() 

            print('Do these records refer to the same thing?')
            valid_response = False
            while not valid_response:
                label = input('(y)es / (n)o / (u)nsure / (f)inished\n')
                if label in ['y', 'n', 'u', 'f']:
                    valid_response = True

            if label == 'y' :
                labels['match'].append(record_pair)
                labeled = True
            elif label == 'n' :
                labels['distinct'].append(record_pair)
                labeled = True
            elif label == 'f':
                print('Finished labeling')
                finished = True
            elif label != 'u':
                print('Nonvalid response')
                raise

        if labeled :
            deduper.markPairs(labels)
        

def trainingDataLink(data_1, data_2, common_key, training_size=50000) : # pragma : nocover
    '''
    Construct training data for consumption by the ActiveLearning 
    markPairs method from already linked datasets.
    
    Arguments : 
    data_1        -- Dictionary of records from first dataset, where the keys
                     are record_ids and the values are dictionaries with the 
                     keys being field names

    data_2        -- Dictionary of records from second dataset, same form as 
                     data_1
    
    common_key    -- The name of the record field that uniquely identifies 
                     a match
    
    training_size -- the rough limit of the number of training examples, 
                     defaults to 50000
    
    Warning:
    
    Every match must be identified by the sharing of a common key. 
    This function assumes that if two records do not share a common key 
    then they are distinct records. 
    '''
    
    
    identified_records = collections.defaultdict(lambda: [[],[]])
    matched_pairs = set()
    distinct_pairs = set()

    for record_id, record in data_1.items() :
        identified_records[record[common_key]][0].append(record_id)

    for record_id, record in data_2.items() :
        identified_records[record[common_key]][1].append(record_id)

    for keys_1, keys_2 in identified_records.values() :
        if keys_1 and keys_2 :
            matched_pairs.update(itertools.product(keys_1, keys_2))

    distinct_pairs = set(itertools.product(data_1.keys(), data_2.keys()))
    distinct_pairs -= matched_pairs
    distinct_pairs = random.sample(distinct_pairs, training_size)

    matched_records = [(data_1[key_1], data_2[key_2])
                       for key_1, key_2 in matched_pairs]

    distinct_records = [(data_1[key_1], data_2[key_2])
                        for key_1, key_2 in distinct_pairs]

    training_pairs = {'match' : matched_records, 
                      'distinct' : distinct_records} 

    return training_pairs        
        
        
def trainingDataDedupe(data, common_key, training_size=50000) : # pragma : nocover
    '''
    Construct training data for consumption by the ActiveLearning 
    markPairs method from an already deduplicated dataset.
    
    Arguments : 
    data          -- Dictionary of records, where the keys are record_ids and 
                     the values are dictionaries with the keys being 
                     field names

    common_key    -- The name of the record field that uniquely identifies 
                     a match
    
    training_size -- the rough limit of the number of training examples, 
                     defaults to 50000
    
    Warning:
    
    Every match must be identified by the sharing of a common key. 
    This function assumes that if two records do not share a common key 
    then they are distinct records. 
    '''

    
    identified_records = collections.defaultdict(list)
    matched_pairs = set()
    distinct_pairs = set()
    unique_record_ids = set()
    
    # a list of record_ids associated with each common_key
    for record_id, record in data.items() :
        unique_record_ids.add(record_id)
        identified_records[record[common_key]].append(record_id)

    # all combinations of matched_pairs from each common_key group
    for record_ids in identified_records.values() :
        if len(record_ids) > 1 :
            matched_pairs.update(itertools.combinations(sorted(record_ids), 2))

    # calculate indices using dedupe.core.randomPairs to avoid 
    # the memory cost of enumerating all possible pairs
    unique_record_ids = list(unique_record_ids)
    pair_indices = randomPairs(len(unique_record_ids), training_size)
    distinct_pairs = set()
    for i, j in pair_indices:
        distinct_pairs.add((unique_record_ids[i],
                            unique_record_ids[j]))

    distinct_pairs -= matched_pairs

    matched_records = [(data[key_1], data[key_2])
                       for key_1, key_2 in matched_pairs]

    distinct_records = [(data[key_1], data[key_2])
                        for key_1, key_2 in distinct_pairs]

    training_pairs = {'match' : matched_records, 
                      'distinct' : distinct_records} 

    return training_pairs


def canonicalize(record_cluster): # pragma : nocover
    """
    Constructs a canonical representation of a duplicate cluster by
    finding canonical values for each field

    Arguments:
    record_cluster     --A list of records within a duplicate cluster, where 
                         the records are dictionaries with field 
                         names as keys and field values as values

    """
    return getCanonicalRep(record_cluster)
