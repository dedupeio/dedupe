#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools

import numpy
import numpy.lib.recfunctions as rfn
import fastcluster
import hcluster

def connected_components(edgelist) :
    root = {}
    component = {}
    component_edges = {}
    for edge in edgelist :
        (a, b), weight = edge
        edge = ((a, b), weight) # turn numpy arrays back 
                                #into simple Python objects
        root_a = root.get(a)
        root_b = root.get(b)

        if root_a is None and root_b is None :
            component[a] = set([a, b])
            component_edges[a] = set([edge])
            root[a] = root[b] = a
        elif root_a is None or root_b is None :
            if root_a is None :
                a, b = b, a
                root_a, root_b = root_b, root_a
            component[root_a].add(b)
            component_edges[root_a].add(edge)
            root[b] = root_a
        elif root_a != root_b :
            component_a = component[root_a]
            component_b = component[root_b]
            if len(component_a) < len(component_b) :
                root_a, root_b = root_b, root_a
                component_a, component_b = component_b, component_a

            component_a |= component_b
            component_edges[root_a] |= component_edges[root_b]
            for node in component_b :
                root[node] = root_a

            del component[root_b]
            del component_edges[root_b]
        else : 
            component_edges[root_a].add(edge)

    return [list(sub_graph) for sub_graph in component_edges.values()]

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

    dupe_sub_graphs = connected_components(dupes)

    clustering = {}
    cluster_id = 0
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:
            pairs = numpy.array(sub_graph, dupes.dtype)

            (i_to_id, condensed_distances) = condensedDistance(pairs)
            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid', 
                                          preserve_input=False)

            partition = hcluster.fcluster(linkage, 
                                          threshold,
                                          criterion='distance')

            for (i, sub_cluster_id) in enumerate(partition):
                clustering.setdefault(cluster_id + sub_cluster_id, []).append(i_to_id[i])
            cluster_id += max(partition) + 1
        else:

            clustering[cluster_id] = sub_graph[0][0]
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

def gazetteMatching(dupes, threshold=0.5):
    covered_vertex_A = set([])
    clusters = []

    sorted_dupes = sorted(dupes, key=lambda score: score[1], reverse=True)
    dupes_list = [dupe for dupe in sorted_dupes if dupe[1] >= threshold]

    for dupe in dupes_list:
        vertices = dupe[0]
        if vertices[0] not in covered_vertex_A:
            clusters.append(vertices)
            covered_vertex_A.update([vertices[0]])

    return clusters
