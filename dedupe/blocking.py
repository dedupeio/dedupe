#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import product, chain, combinations
from math import sqrt, log
import core
from random import sample


def blockingIndex(data_d, predicates):
    blocked_data = defaultdict(set)
    for (key, instance) in data_d.items():
        for predicate in predicates:
            predicate_tuples = product(*[F(data_d[key][field])
                                         for (F, field) in predicate]
                                       )

            for predicate_tuple in predicate_tuples:
                blocked_data[str(predicate_tuple)].add(key)

    return blocked_data


def mergeBlocks(blocked_data):
    candidates = set()
    for block in blocked_data.values():
        if len(block) > 1:
            block = sorted(block)
            for pair in combinations(block, 2):
                candidates.add(pair)

    return candidates


def allCandidates(data_d):
    return list(combinations(sorted(data_d.keys()), 2))


def semiSupervisedNonDuplicates(data_d, data_model,
                                nonduplicate_confidence_threshold=.7):

    # this is an expensive call and we're making it multiple times
    pairs = allCandidates(data_d)
    record_distances = core.recordDistances(pairs, data_d, data_model)

    confident_nondupes_ids = []
    scored_pairs = core.scorePairs(record_distances, data_model)

    for (i, score) in enumerate(scored_pairs):
        if score < 1 - nonduplicate_confidence_threshold:
            confident_nondupes_ids.append(record_distances['pairs'][i])

    confident_nondupes_pairs = [(data_d[pair[0]], data_d[pair[1]])
                                for pair in
                                confident_nondupes_ids]

    return confident_nondupes_pairs


class Blocking:

    def __init__(self,
                 training_pairs,
                 predicate_functions,
                 data_model,
                 eta=1,
                 epsilon=1,
                 ):
        self.epsilon = epsilon
        self.predicate_functions = predicate_functions

        self.fields = [field for field in data_model['fields']
                       if data_model['fields'][field]['type']
                       != 'Interaction']

        n_sample_distinct = 1000
        if len(training_pairs[0]) <= n_sample_distinct:
            self.training_distinct = (training_pairs[0])[:]
        else:
            self.training_distinct = sample((training_pairs[0])[:],
                                            n_sample_distinct)

        self.training_dupes = (training_pairs[1])[:]

        # We want to throw away the predicates that puts together too many
        # distinct pairs
        sample_size = len(self.training_dupes) \
            + len(self.training_distinct)
        self.coverage_threshold = eta * sample_size

    # Approximate learning of blocking following the ApproxRBSetCover from
    # page 102 of Bilenko
    def trainBlocking(self, disjunctive=True):
        self.predicate_set = self.createPredicateSet(disjunctive)
        n_predicates = len(self.predicate_set)

        found_dupes = self.predicateCoverage(self.training_dupes)
        found_distinct = self.predicateCoverage(self.training_distinct)

        # Only consider predicates that cover at least one duplicate pair
        self.predicate_set = found_dupes.keys()

        # We want to throw away the predicates that puts together too
        # many distinct pairs
        [self.predicate_set.remove(predicate)
         for predicate
         in found_distinct
         if len(found_distinct[predicate]) >= self.coverage_threshold]

        # Expected number of predicates that should cover a duplicate
        # pair
        expected_dupe_cover = sqrt(n_predicates
                                   / log(len(self.training_dupes)))

        found_distinct = self.filterOutIndistinctPairs(expected_dupe_cover,
                                                       found_distinct,
                                                       self.training_distinct)

        final_predicate_set = self.findOptimumBlocking(self.training_dupes,
                                                       self.predicate_set,
                                                       found_dupes,
                                                       found_distinct)

        print 'Final predicate set'
        print final_predicate_set

        if final_predicate_set:
            return final_predicate_set
        else:
            print 'No predicate found!'
            raise

    def predicateCoverage(self, pairs):
        coverage = defaultdict(list)
        for pair in pairs:
            for predicate in self.predicate_set:
                keys1 = set(product(*[F(pair[0][field]) for (F,
                            field) in predicate]))
                keys2 = set(product(*[F(pair[1][field]) for (F,
                            field) in predicate]))
                if keys1 & keys2:
                    coverage[predicate].append(pair)

        return coverage

    def createPredicateSet(self, disjunctive):

        # The set of simple predicates
        predicate_set = list(product(self.predicate_functions,
                                     self.fields))

        if disjunctive:
            disjunctive_predicates = list(combinations(predicate_set, 2))

            # filter out disjunctive predicates that operate on same
            # field
            disjunctive_predicates = [predicate for predicate
                                      in disjunctive_predicates
                                      if predicate[0][1]
                                      != predicate[1][1]]

            predicate_set = [(predicate, ) for predicate in
                             predicate_set]
            predicate_set.extend(disjunctive_predicates)
        else:

            predicate_set = [(predicate, ) for predicate in
                             predicate_set]

        return predicate_set

    def filterOutIndistinctPairs(self,
                                 expected_dupe_cover,
                                 found_distinct,
                                 training_distinct,
                                 ):

        # We don't want to penalize a blocker if it puts distinct
        # pairs together that look like they could be duplicates.
        predicate_count = defaultdict(int)
        for pair in chain(*found_distinct.values()):
            predicate_count[pair] += 1

        training_distinct = [pair for pair in training_distinct
                             if predicate_count[pair]
                             < expected_dupe_cover]

        return self.predicateCoverage(training_distinct)

    def findOptimumBlocking(self,
                            training_dupes,
                            predicate_set,
                            found_dupes,
                            found_distinct,
                            ):

        # Greedily find the predicates that, at each step, covers the
        # most duplicates and covers the least distinct pairs, due to
        # Chvatal, 1979
        final_predicate_set = []
        n_training_dupes = len(training_dupes)
        print 'Uncovered dupes'
        print n_training_dupes
        while n_training_dupes >= self.epsilon:

            optimumCover = 0
            bestPredicate = None
            for predicate in predicate_set:
                try:
                    cover = (len(found_dupes[predicate])
                             / float(len(found_distinct[predicate]))
                             )
                except ZeroDivisionError:
                    cover = len(found_dupes[predicate])

                if cover > optimumCover:
                    optimumCover = cover
                    bestPredicate = predicate

            if not bestPredicate:
                print 'Ran out of predicates'
                break

            predicate_set.remove(bestPredicate)
            n_training_dupes -= len(found_dupes[bestPredicate])
            [training_dupes.remove(pair) for pair in
             found_dupes[bestPredicate]]
            found_dupes = self.predicateCoverage(training_dupes)

            final_predicate_set.append(bestPredicate)

        return final_predicate_set
