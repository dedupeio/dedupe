#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools

import numpy
import fastcluster
import hcluster
import networkx
from networkx.algorithms.components.connected import connected_components


def condensedDistance(dupes):
    '''
    Convert the pairwise list of distances in dupes to "condensed
    distance matrix" required by the hierarchical clustering
    algorithms. Also return a dictionary that maps the distance matrix
    to the record_ids.
   
    The condensed distance matrix is described in the scipy
    documentation of scipy.cluster.hierarchy.linkage
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    '''

    candidate_set = numpy.unique(dupes['pairs'])
    candidate_set = numpy.sort(candidate_set)

    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes['pairs'])
    id_1 = ids[:, 0]
    id_2 = ids[:, 1]

    N = len(numpy.union1d(id_1, id_2))
    matrix_length = N * (N - 1) / 2

    step = (N - id_1) * (N - id_1 - 1) / 2
    index = matrix_length - step + id_2 - id_1 - 1

    condensed_distances = numpy.ones(matrix_length, 'f4')
    condensed_distances[index] = 1 - dupes['score']

    return (i_to_id, condensed_distances)


def cluster(dupes, id_type, threshold=.5):
    '''
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    Keyword arguments:
    threshold -- number betweent 0 and 1 (default is .5). lowering the 
                 number will increase precision, raising it will increase
                 recall
    '''

    threshold = 1 - threshold

    score_dtype = [('pairs', id_type, 2), ('score', 'f4', 1)]

    dupe_graph = networkx.Graph()
    dupe_graph.add_weighted_edges_from((x[0], x[1], y) for (x, y) in dupes)

    dupe_sub_graphs = connected_components(dupe_graph)

    clustering = {}
    cluster_id = 0
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 2:
            pair_gen = ((sorted(x[0:2]), x[2]['weight'])
                        for x in dupe_graph.edges_iter(sub_graph, data=True))

            pairs = numpy.fromiter(pair_gen, dtype=score_dtype)

            (i_to_id, condensed_distances) = condensedDistance(pairs)
            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid', 
                                          preserve_input=False)

            partition = hcluster.fcluster(linkage, 
                                          threshold,
                                          criterion='distance')

            for (i, sub_cluster_id) in enumerate(partition):
                clustering.setdefault(cluster_id + sub_cluster_id, []).append(i_to_id[i])
            
            cluster_id += max(partition)
        else:

            clustering[cluster_id] = sub_graph
            cluster_id += 1

    clusters = [set(l) for l in clustering.values() if len(l) > 1]

    return clusters
