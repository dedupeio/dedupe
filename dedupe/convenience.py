#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from builtins import input

import collections
import itertools
import sys
from dedupe.core import randomPairs, randomPairsMatch
from dedupe.canonical import getCanonicalRep


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def consoleLabel(deduper):  # pragma: no cover
    '''
    Command line interface for presenting and labeling training pairs
    by the user

    Argument :
    A deduper object
    '''

    finished = False
    use_previous = False
    fields = unique(field.field
                    for field
                    in deduper.data_model.primary_fields)

    buffer_len = 1  # Max number of previous operations
    examples_buffer = []
    uncertain_pairs = []

    while not finished:
        if use_previous:
            record_pair, _ = examples_buffer.pop(0)
            use_previous = False
        else:
            if not uncertain_pairs:
                uncertain_pairs = deduper.uncertainPairs()

            try:
                record_pair = uncertain_pairs.pop()
            except IndexError:
                break

        n_match = (len(deduper.training_pairs['match']) +
                   sum(label == 'match' for _, label in examples_buffer))
        n_distinct = (len(deduper.training_pairs['distinct']) +
                      sum(label == 'distinct' for _, label in examples_buffer))

        for pair in record_pair:
            for field in fields:
                line = "%s : %s" % (field, pair[field])
                print(line, file=sys.stderr)
            print(file=sys.stderr)

        print("{0}/10 positive, {1}/10 negative".format(n_match, n_distinct),
              file=sys.stderr)
        print('Do these records refer to the same thing?', file=sys.stderr)

        valid_response = False
        user_input = ''
        while not valid_response:
            if examples_buffer:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished / (p)revious'
                valid_responses = {'y', 'n', 'u', 'f', 'p'}
            else:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished'
                valid_responses = {'y', 'n', 'u', 'f'}

            print(prompt, file=sys.stderr)
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == 'y':
            examples_buffer.insert(0, (record_pair, 'match'))
        elif user_input == 'n':
            examples_buffer.insert(0, (record_pair, 'distinct'))
        elif user_input == 'u':
            examples_buffer.insert(0, (record_pair, 'uncertain'))
        elif user_input == 'f':
            print('Finished labeling', file=sys.stderr)
            finished = True
        elif user_input == 'p':
            use_previous = True
            uncertain_pairs.append(record_pair)

        if len(examples_buffer) > buffer_len:
            record_pair, label = examples_buffer.pop()
            if label in ['distinct', 'match']:
                examples = {'distinct': [], 'match': []}
                examples[label].append(record_pair)
                deduper.markPairs(examples)

    for record_pair, label in examples_buffer:
        if label in ['distinct', 'match']:
            examples = {'distinct': [], 'match': []}
            examples[label].append(record_pair)
            deduper.markPairs(examples)


def trainingDataLink(data_1, data_2, common_key, training_size=50000):  # pragma: nocover
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

    identified_records = collections.defaultdict(lambda: [[], []])
    matched_pairs = set()
    distinct_pairs = set()

    for record_id, record in data_1.items():
        identified_records[record[common_key]][0].append(record_id)

    for record_id, record in data_2.items():
        identified_records[record[common_key]][1].append(record_id)

    for keys_1, keys_2 in identified_records.values():
        if keys_1 and keys_2:
            matched_pairs.update(itertools.product(keys_1, keys_2))

    keys_1 = list(data_1.keys())
    keys_2 = list(data_2.keys())

    random_pairs = [(keys_1[i], keys_2[j])
                    for i, j
                    in randomPairsMatch(len(data_1), len(data_2),
                                        training_size)]

    distinct_pairs = (
        pair for pair in random_pairs if pair not in matched_pairs)

    matched_records = [(data_1[key_1], data_2[key_2])
                       for key_1, key_2 in matched_pairs]
    distinct_records = [(data_1[key_1], data_2[key_2])
                        for key_1, key_2 in distinct_pairs]

    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs


def trainingDataDedupe(data, common_key, training_size=50000):  # pragma: nocover
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
    for record_id, record in data.items():
        unique_record_ids.add(record_id)
        identified_records[record[common_key]].append(record_id)

    # all combinations of matched_pairs from each common_key group
    for record_ids in identified_records.values():
        if len(record_ids) > 1:
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

    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs


def canonicalize(record_cluster):  # pragma: nocover
    """
    Constructs a canonical representation of a duplicate cluster by
    finding canonical values for each field

    Arguments:
    record_cluster     --A list of records within a duplicate cluster, where
                         the records are dictionaries with field
                         names as keys and field values as values

    """
    return getCanonicalRep(record_cluster)
