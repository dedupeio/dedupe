#!/usr/bin/python
from __future__ import annotations

import collections
import itertools
import random
import sys
import warnings
from typing import Iterator, Literal, Tuple, overload

import numpy

import dedupe
from dedupe._typing import (
    DataInt,
    DataStr,
    RecordDict,
    RecordDictPair,
    RecordID,
    TrainingData,
)
from dedupe.canonical import getCanonicalRep
from dedupe.core import unique

IndicesIterator = Iterator[Tuple[int, int]]


def randomPairs(n_records: int, sample_size: int) -> IndicesIterator:
    """
    Return random combinations of indices for a square matrix of size n
    records. For a discussion of how this works see
    http://stackoverflow.com/a/14839010/98080

    """
    n: int = n_records * (n_records - 1) // 2

    if not sample_size:
        return iter([])
    elif sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        try:
            random_pairs = numpy.array(
                random.sample(range(n), sample_size), dtype=numpy.uint64
            )
        except OverflowError:
            return randomPairsWithReplacement(n_records, sample_size)

    b: int = 1 - 2 * n_records

    i = (-b - 2 * numpy.sqrt(2 * (n - random_pairs) + 0.25)) // 2
    i = i.astype(numpy.int64)

    j = random_pairs + i * (b + i + 2) // 2 + 1
    j = j.astype(numpy.uint64)

    return zip(i, j)


def randomPairsMatch(
    n_records_A: int, n_records_B: int, sample_size: int
) -> IndicesIterator:
    """
    Return random combinations of indices for record list A and B
    """
    n: int = n_records_A * n_records_B

    if not sample_size:
        return iter([])
    elif sample_size >= n:
        random_pairs = numpy.arange(n)
    else:
        random_pairs = numpy.array(random.sample(range(n), sample_size))

    i, j = numpy.unravel_index(random_pairs, (n_records_A, n_records_B))

    return zip(i, j)


def randomPairsWithReplacement(n_records: int, sample_size: int) -> IndicesIterator:
    # If the population is very large relative to the sample
    # size than we'll get very few duplicates by chance
    warnings.warn("The same record pair may appear more than once in the sample")

    try:
        random_indices = numpy.random.randint(n_records, size=sample_size * 2)
    except (OverflowError, ValueError):
        max_int: int = numpy.iinfo("int").max
        warnings.warn(
            "Asked to sample pairs from %d records, will only sample pairs from first %d records"
            % (n_records, max_int)
        )

        random_indices = numpy.random.randint(max_int, size=sample_size * 2)

    random_indices = random_indices.reshape((-1, 2))
    random_indices.sort(axis=1)

    return ((p.item(), q.item()) for p, q in random_indices)


def _print(*args) -> None:
    print(*args, file=sys.stderr)


LabeledPair = Tuple[RecordDictPair, Literal["match", "distinct", "unsure"]]


def _mark_pair(deduper: dedupe.api.ActiveMatching, labeled_pair: LabeledPair) -> None:
    record_pair, label = labeled_pair
    examples: TrainingData = {"distinct": [], "match": []}
    if label == "unsure":
        # See https://github.com/dedupeio/dedupe/issues/984 for reasoning
        examples["match"].append(record_pair)
        examples["distinct"].append(record_pair)
    else:
        # label is either "match" or "distinct"
        examples[label].append(record_pair)
    deduper.mark_pairs(examples)


def console_label(deduper: dedupe.api.ActiveMatching) -> None:  # pragma: no cover
    """
    Train a matcher instance (Dedupe, RecordLink, or Gazetteer) from the command line.
    Example

    .. code:: python

       > deduper = dedupe.Dedupe(variables)
       > deduper.prepare_training(data)
       > dedupe.console_label(deduper)
    """

    finished = False
    use_previous = False
    fields = unique(var.field for var in deduper.data_model.field_variables)

    buffer_len = 1  # Max number of previous operations
    unlabeled: list[RecordDictPair] = []
    labeled: list[LabeledPair] = []

    n_match = len(deduper.training_pairs["match"])
    n_distinct = len(deduper.training_pairs["distinct"])

    while not finished:
        if use_previous:
            record_pair, label = labeled.pop(0)
            if label == "match":
                n_match -= 1
            elif label == "distinct":
                n_distinct -= 1
            use_previous = False
        else:
            try:
                if not unlabeled:
                    unlabeled = deduper.uncertain_pairs()

                record_pair = unlabeled.pop()
            except IndexError:
                break

        for record in record_pair:
            for field in fields:
                line = "{} : {}".format(field, record[field])
                _print(line)
            _print()
        _print(f"{n_match}/10 positive, {n_distinct}/10 negative")
        _print("Do these records refer to the same thing?")

        valid_response = False
        user_input = ""
        while not valid_response:
            if labeled:
                _print("(y)es / (n)o / (u)nsure / (f)inished / (p)revious")
                valid_responses = {"y", "n", "u", "f", "p"}
            else:
                _print("(y)es / (n)o / (u)nsure / (f)inished")
                valid_responses = {"y", "n", "u", "f"}
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == "y":
            labeled.insert(0, (record_pair, "match"))
            n_match += 1
        elif user_input == "n":
            labeled.insert(0, (record_pair, "distinct"))
            n_distinct += 1
        elif user_input == "u":
            labeled.insert(0, (record_pair, "unsure"))
        elif user_input == "f":
            _print("Finished labeling")
            finished = True
        elif user_input == "p":
            use_previous = True
            unlabeled.append(record_pair)

        while len(labeled) > buffer_len:
            _mark_pair(deduper, labeled.pop())

    for labeled_pair in labeled:
        _mark_pair(deduper, labeled_pair)


@overload
def training_data_link(
    data_1: DataInt, data_2: DataInt, common_key: str, training_size: int = 50000
) -> TrainingData:  # pragma: nocover
    ...


@overload
def training_data_link(
    data_1: DataStr, data_2: DataStr, common_key: str, training_size: int = 50000
) -> TrainingData:  # pragma: nocover
    ...


def training_data_link(
    data_1, data_2, common_key, training_size=50000
) -> TrainingData:  # pragma: nocover
    """
    Construct training data for consumption by the func:`mark_pairs`
    method from already linked datasets.

    Args:

        data_1: Dictionary of records from first dataset, where the
                keys are record_ids and the values are dictionaries
                with the keys being field names
        data_2: Dictionary of records from second dataset, same form as
                data_1
        common_key: The name of the record field that uniquely identifies
                    a match
        training_size: the rough limit of the number of training examples,
                       defaults to 50000

    .. note::

         Every match must be identified by the sharing of a common key.
         This function assumes that if two records do not share a common key
         then they are distinct records.
    """

    identified_records: dict[str, tuple[list[RecordID], list[RecordID]]]
    identified_records = collections.defaultdict(lambda: ([], []))
    matched_pairs: set[tuple[RecordID, RecordID]] = set()
    distinct_pairs: set[tuple[RecordID, RecordID]] = set()

    for record_id, record in data_1.items():
        identified_records[record[common_key]][0].append(record_id)

    for record_id, record in data_2.items():
        identified_records[record[common_key]][1].append(record_id)

    for keys_1, keys_2 in identified_records.values():
        if keys_1 and keys_2:
            matched_pairs.update(itertools.product(keys_1, keys_2))

    keys_1 = list(data_1.keys())
    keys_2 = list(data_2.keys())

    random_pairs = [
        (keys_1[i], keys_2[j])
        for i, j in randomPairsMatch(len(data_1), len(data_2), training_size)
    ]

    distinct_pairs = {pair for pair in random_pairs if pair not in matched_pairs}

    matched_records = [(data_1[key_1], data_2[key_2]) for key_1, key_2 in matched_pairs]
    distinct_records = [
        (data_1[key_1], data_2[key_2]) for key_1, key_2 in distinct_pairs
    ]

    training_pairs: TrainingData
    training_pairs = {"match": matched_records, "distinct": distinct_records}

    return training_pairs


@overload
def training_data_dedupe(
    data: DataInt, common_key: str, training_size: int = 50000
) -> TrainingData:  # pragma: nocover
    ...


@overload
def training_data_dedupe(
    data: DataStr, common_key: str, training_size: int = 50000
) -> TrainingData:  # pragma: nocover
    ...


def training_data_dedupe(
    data, common_key, training_size=50000
) -> TrainingData:  # pragma: nocover
    """
    Construct training data for consumption by the func:`mark_pairs`
    method from an already deduplicated dataset.

    Args:

        data: Dictionary of records where the keys are record_ids and
              the values are dictionaries with the keys being field names
        common_key: The name of the record field that uniquely identifies
                    a match
        training_size: the rough limit of the number of training examples,
                       defaults to 50000

    .. note::

         Every match must be identified by the sharing of a common key.
         This function assumes that if two records do not share a common key
         then they are distinct records.
    """

    identified_records: dict[str, list[RecordID]]
    identified_records = collections.defaultdict(list)
    matched_pairs: set[tuple[RecordID, RecordID]] = set()
    distinct_pairs: set[tuple[RecordID, RecordID]] = set()
    unique_record_ids: set[RecordID] = set()

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
    unique_record_ids_l = list(unique_record_ids)
    pair_indices = randomPairs(len(unique_record_ids), training_size)
    distinct_pairs = set()
    for i, j in pair_indices:
        distinct_pairs.add((unique_record_ids_l[i], unique_record_ids_l[j]))

    distinct_pairs -= matched_pairs

    matched_records = [(data[key_1], data[key_2]) for key_1, key_2 in matched_pairs]

    distinct_records = [(data[key_1], data[key_2]) for key_1, key_2 in distinct_pairs]

    training_pairs: TrainingData
    training_pairs = {"match": matched_records, "distinct": distinct_records}

    return training_pairs


def canonicalize(record_cluster: list[RecordDict]) -> RecordDict:  # pragma: nocover
    """
    Constructs a canonical representation of a duplicate cluster by
    finding canonical values for each field

    Args:
        record_cluster: A list of records within a duplicate cluster, where
                        the records are dictionaries with field
                        names as keys and field values as values

    """
    return getCanonicalRep(record_cluster)
