#!/usr/bin/python
# -*- coding: utf-8 -*-
from future.utils import viewitems

import itertools

import warnings
import numpy
import fastcluster
import hcluster

def connected_components(edgelist, max_components) :

    root = {}
    component = {}
    indices = {}

    if len(edgelist['pairs']) == 0:
        raise StopIteration()

    it = numpy.nditer(edgelist['pairs'], ['external_loop'])

    for i, (a,b) in enumerate(it) :
        root_a = root.get(a)
        root_b = root.get(b)

        if root_a is None and root_b is None :
            component[a] = {a, b}
            indices[a] = [i]
            root[a] = root[b] = a
        elif root_a is None or root_b is None :
            if root_a is None :
                a, b = b, a
                root_a, root_b = root_b, root_a
            component[root_a].add(b)
            indices[root_a].append(i)
            root[b] = root_a
        elif root_a != root_b :
            component_a = component[root_a]
            component_b = component[root_b]
            if len(component_a) < len(component_b) :
                root_a, root_b = root_b, root_a
                component_a, component_b = component_b, component_a

            component_a |= component_b
            indices[root_a].extend(indices[root_b])
            indices[root_a].append(i)

            for node in component_b :
                root[node] = root_a

            del component[root_b]
            del indices[root_b]
        else : 
            indices[root_a].append(i)

    for root in component :
        n_components = len(component[root])
        sub_graph = edgelist[indices[root]]
        
        if n_components > max_components :
            threshold = numpy.min(sub_graph['score'])
            threshold *= 1.1 
            warnings.warn('A component contained %s elements. '
                          'Components larger than %s are '
                          're-filtered. The threshold for this '
                          'filtering is %s' % (n_components, 
                                               max_components,
                                               threshold)) 
            filtered_sub_graph = sub_graph[sub_graph['score'] > threshold]	
            for sub_graph in connected_components(filtered_sub_graph, 
                                                  max_components) :
               yield sub_graph
        else :
            yield sub_graph
     

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
    condensed_distances[index.astype(int)] = 1 - dupes['score']


    return i_to_id, condensed_distances, N


def cluster(dupes, threshold=.5, max_components=30000):
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

    dupe_sub_graphs = connected_components(dupes, max_components)

    clustering = {}
    cluster_id = 0
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:

            i_to_id, condensed_distances, N = condensedDistance(sub_graph)

            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid', 
                                          preserve_input=False)

            partition = hcluster.fcluster(linkage, 
                                          threshold,
                                          criterion='distance')

            clusters = {}

            for (i, sub_cluster_id) in enumerate(partition):
                clusters.setdefault(cluster_id + sub_cluster_id, []).append(i)

            for cluster_id, items in viewitems(clusters) :
                if len(items) > 1 :
                    scores = confidences(items, condensed_distances, N)
                    clustering[cluster_id] =\
                        (tuple(i_to_id[item] for item in items), tuple(scores))

            cluster_id += max(partition) + 1
        else:
            ids, score = sub_graph[0]
            clustering[cluster_id] = (tuple(ids), tuple([score]*2))
            cluster_id += 1
            

    return clustering.values()

def confidences(items, condensed_distances, d) :
    scores = dict.fromkeys(items, 0)
    for i, j in itertools.combinations(items, 2) :
        index = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
        dist = condensed_distances[index]
        scores[i] += dist
        scores[j] += dist
    scores = numpy.array([v for (k, v) in sorted(scores.items())])
    scores /= len(items) - 1
    scores = 1 - scores
    return scores

def greedyMatching(dupes, threshold=0.5):
    dupes = numpy.array(dupes)
    covered_vertex_A = set()
    covered_vertex_B = set()
    clusters = []

    sorted_dupes = sorted(dupes, key=lambda score: score[1], reverse=True)
    dupes_list = (dupe for dupe in sorted_dupes if dupe[1] >= threshold)

    for vertices, score in dupes_list:
        a, b = vertices
        if a not in covered_vertex_A and b not in covered_vertex_B:
            clusters.append((vertices, score))
            covered_vertex_A.add(a)
            covered_vertex_B.add(b)

    return clusters

def gazetteMatching(dupes, threshold=0.5, n_matches=1):
    dupes = numpy.array(dupes) 
    clusters = []

    sorted_dupes = sorted(dupes, key=lambda pair: (pair[0][0], -pair[1]))
    dupes_list = [dupe for dupe in sorted_dupes if dupe[1] >= threshold]

    if dupes_list :
        group = dupes_list[0][0][0]
        matches = []
        i = 0

        for pair, score in dupes_list:
            a, b = pair
            if a == group :
                if i < n_matches :
                    matches.append((pair, score))
                    i += 1
            else :
                clusters.append(tuple(matches))
                matches = [(pair, score)]
                i = 1
                group = a

        clusters.append(tuple(matches))

    return clusters
