#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import combinations


def neighborDict(duplicates):
    neighbors = defaultdict(list)

    for (pair, similarity) in duplicates:
        (candidate_1, candidate_2) = pair
        distance = 1 - similarity

        neighbors[candidate_1].append((candidate_2, distance))
        neighbors[candidate_2].append((candidate_1, distance))

    for (candidate, neighborhood) in neighbors.iteritems():
        neighborhood += [(candidate, 0)]
        neighborhood = sorted(neighborhood,
                              key=lambda neighborhood: neighborhood[0])
        neighborhood = sorted(neighborhood,
                              key=lambda neighborhood: neighborhood[1])
        neighbors[candidate] = neighborhood

    return neighbors


def neighborhoodGrowth(distances, neighborhood_multiplier):
    smallest_distance = min(distances)
    neighborhood_growth = sum([distance <= neighborhood_multiplier
                              * smallest_distance for distance in
                              distances])

    return neighborhood_growth


def kOverlap(neighborhood_1, neighborhood_2):

    K = min(len(neighborhood_1), len(neighborhood_2))
    overlap = [False] * K

    for k in xrange(1, K + 1):
        if set(neighborhood_1[:k]) == set(neighborhood_2[:k]):
            overlap[k - 1] = True

    return overlap


def compactPairs(neighbors, neighborhood_multiplier):

    compact_pairs = []

    candidates = neighbors.keys()
    candidates = sorted(candidates)
    candidate_pairs = combinations(candidates, 2)

    for pair in candidate_pairs:
        (candidate_1, candidate_2) = pair

        neighbors_1 = neighbors[candidate_1]
        neighbors_2 = neighbors[candidate_2]

        (neighbor_ids_1, distances_1) = zip(*neighbors_1)
        (neighbor_ids_2, distances_2) = zip(*neighbors_2)

        if (candidate_1 in neighbor_ids_2) and (candidate_2
                                                in neighbor_ids_1):
            k_set_overlap = kOverlap(neighbor_ids_1, neighbor_ids_2)

            if any(k_set_overlap):

        # Since the nearest neighbor to a candidate is always itself the
        # first elements will never overlap

                k_set_overlap = k_set_overlap[1:]

                growths = (neighborhoodGrowth(distances_1[1:],
                           neighborhood_multiplier),
                           neighborhoodGrowth(distances_2[1:],
                           neighborhood_multiplier))

                compact_pairs.append((pair, k_set_overlap, growths))

    return compact_pairs


def partition(compact_pairs, sparseness_threshold):

    assigned_candidates = set([])
    clusters = []

    groups = defaultdict(list)
    for pair in compact_pairs:
        groups[pair[0][0]].append(pair)

    for (group_id, group) in groups.iteritems():
        if group_id not in assigned_candidates:
            (pair_ids, k_compact_set, growths) = zip(*group)

      # Find the largest compact set that is has an aggregate NG below
      # the the threshold

            largest_compact_sets = []
            max_c_set = set([])
            for compact_bools in zip(*k_compact_set):
                max_growth = 0
                c_set = set([group_id])
                for (i, compact_bool) in enumerate(compact_bools):
                    if compact_bool:
                        try:
                            c_set.update([pair_ids[i][1]])
                        except:
                            print c_set
                            raise
                        max_growth = max(max_growth, max(growths[i]))
                if len(c_set) > len(max_c_set) and (max_growth
                                                    <= sparseness_threshold):
                    max_c_set = c_set

            if len(max_c_set) > 1:
                clusters.append(max_c_set)
                assigned_candidates.update(max_c_set)

    return clusters


def growthDistributions(neighbors, neighborhood_multiplier):
    growths = []

    for neighborhood in neighbors.values():
        distances = (zip(*neighborhood)[1])[1:]
        growths.append(neighborhoodGrowth(distances,
                       neighborhood_multiplier))

    distribution = [(growths.count(growth), growth) for growth in
                    set(growths)]

    distribution = sorted(distribution, key=lambda growth: growth[1])

    distribution = [(growth[0] / float(len(growths)), growth[1])
                    for growth in distribution]

    cumulative_distribution = []
    cumulative_growth = 0
    for (i, growth) in enumerate(distribution):
        cumulative_growth += growth[0]
        cumulative_distribution.append((cumulative_growth, growth[1]))

    # print 'Distribution of Growths'
    # for quantile in distribution:
    #     print ('%.2f' % quantile[0], quantile[1])

    return (distribution, cumulative_distribution)


def sparsenessThreshold(neighbors,
                        estimated_dupe_fraction,
                        epsilon=0.1,
                        neighborhood_multiplier=2,
                        ):

    (distribution,
     cumulative_distribution) = growthDistributions(neighbors,
                                                    neighborhood_multiplier)

    growth_quantiles = zip(*cumulative_distribution)[0]

    fraction_window = []
    for (i, quantile) in enumerate(growth_quantiles):
        if quantile >= estimated_dupe_fraction + epsilon:
            break
        elif quantile > estimated_dupe_fraction - epsilon:
            fraction_window.append(i)

  # of the quantiles found, return minimum spike

    for j in range(1, len(fraction_window)):
        if 0 < (distribution[fraction_window[j]][0]
                - distribution[fraction_window[j - 1]][0]):
            return distribution[fraction_window[j]][1]

    return distribution[i + 1][1]


def cluster(duplicates,
            sparseness_threshold=4,
            k_nearest_neighbors=6,
            neighborhood_multiplier=2,
            estimated_dupe_fraction=None,
            ):

    neighbors = neighborDict(duplicates)

    if estimated_dupe_fraction:
        sparseness_threshold = sparsenessThreshold(neighbors,
                                                   estimated_dupe_fraction)
        print 'Sparseness Threshold is ', sparseness_threshold

    neighbors = dict([(k, v[:k_nearest_neighbors + 1]) for (k, v)
                      in neighbors.iteritems()])

    compact_pairs = compactPairs(neighbors, neighborhood_multiplier)

    partitions = partition(compact_pairs, sparseness_threshold)

    return partitions
