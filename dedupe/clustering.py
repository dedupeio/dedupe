#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools

import numpy
import fastcluster
import hcluster
import networkx
from networkx.algorithms.components.connected import connected_components
from networkx.algorithms.bipartite.basic import biadjacency_matrix
from networkx.algorithms import bipartite
from networkx import connected_component_subgraphs


def condensedDistance(dupes):
    '''
    Convert the pairwise list of distances in dupes to "condensed
    distance matrix" required by the hierarchical clustering
    algorithms. Also return a dictionary that maps the distance matrix
    to the record_ids.
   
    The formula for an index of the condensed matrix is

    index = {N choose 2}-{N-row choose 2} + (col-row-1)
          = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1
            ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^   
          matrix_length       row_step
    
    where (row,col) is index of an uncondensed square N X N distance matrix.
    
    See http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
    '''

    candidate_set = numpy.unique(dupes['pairs'])
    candidate_set = numpy.sort(candidate_set)

    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes['pairs'])
    row = ids[:, 0]
    col = ids[:, 1]

    N = len(numpy.union1d(row, col))
    matrix_length = N * (N - 1) / 2

    row_step = (N - row) * (N - row - 1) / 2
    index = matrix_length - row_step + col - row - 1

    condensed_distances = numpy.ones(matrix_length, 'f4')
    condensed_distances[index] = 1 - dupes['score']

    return (i_to_id, condensed_distances)


def cluster(dupes, threshold=.5):
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

    dupe_graph = networkx.Graph()
    dupe_graph.add_weighted_edges_from((x[0], x[1], y) for (x, y) in dupes)

    dupe_sub_graphs = connected_components(dupe_graph)

    clustering = {}
    cluster_id = 0
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 2:
            pair_gen = ((sorted(x[0:2]), x[2]['weight'])
                        for x in dupe_graph.edges_iter(sub_graph, data=True))

            pairs = numpy.fromiter(pair_gen, dtype=dupes.dtype)

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


def greedyMatching(dupes, threshold=0.5):
    covered_vertex_A = set([])
    covered_vertex_B = set([])
    clusters = []

    sorted_dupes = sorted(dupes, key=lambda score: score[1], reverse=True)
    dupes_list = [dupe for dupe in sorted_dupes if dupe[1] >= threshold]

    for dupe in dupes_list:
        vertices = dupe[0]
        if vertices[0] not in covered_vertex_A and vertices[1] not in covered_vertex_B:
            clusters.append(vertices)
            covered_vertex_A.update([vertices[0]])
            covered_vertex_B.update([vertices[1]])

    return clusters
