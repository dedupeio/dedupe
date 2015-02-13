#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools

import warnings
import numpy
import fastcluster
import hcluster
from felzenszwalb._felzenszwalb_cy import _felzenszwalb_graph as segment
from igraph import Graph

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
            component[a] = set([a, b])
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
    condensed_distances[index] = 1 - dupes['score']
    

    return (i_to_id, condensed_distances)


def clusterOld(dupes, threshold=.5, max_components=30000):
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

            (i_to_id, condensed_distances) = condensedDistance(sub_graph)
            N = max(i_to_id) + 1

            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid', 
                                          preserve_input=False)

            partition = hcluster.fcluster(linkage, 
                                          threshold,
                                          criterion='distance')

            clusters = {}

            for (i, sub_cluster_id) in enumerate(partition):
                clusters.setdefault(cluster_id + sub_cluster_id, []).append(i)

            distances = hcluster.squareform(condensed_distances)
            
            for cluster_id, items in clusters.iteritems() :
                if len(items) > 1 :
                    scores = confidences(items, distances)
                    clustering[cluster_id] =\
                        (tuple(i_to_id[item] for item in items), tuple(scores))

            cluster_id += max(partition) + 1
        else:
            ids, score = sub_graph[0]
            clustering[cluster_id] = (tuple(ids), tuple([score]*2))
            cluster_id += 1
            

    return clustering.values()

def cluster(dupes, threshold=.5, max_components=30000):
    '''
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold
    '''
    # dupes = [([1004, 2309], 0.9792425036430359) ([3331, 1771], 0.9880577325820923)
    #            ([3331, 2218], 0.9754224419593811) ..., ([1674, 655], 0.7581549882888794)
    #            ([3157, 1718], 0.9362305402755737) ([2372, 2923], 0.7735115885734558)]
    # <class 'numpy.core.memmap.memmap'>

    # set felsenszwalb parameters here. how to incorporate threshold?
    scale_param = 3
    smoothing_param = 1

    all_clusters = []

    dupe_sub_graphs = connected_components(dupes, max_components)

    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:

            # ####################################
            # ########### felzenszwalb ###########
            # ####################################
            # candidate_set = numpy.unique(sub_graph['pairs'])
            # candidate_set = numpy.sort(candidate_set)

            # (i_to_id, condensed_distances) = condensedDistance(sub_graph)
            # distances = hcluster.squareform(condensed_distances)

            # edges = candidate_set.searchsorted(sub_graph['pairs'])
            # costs = []
            # for i, dupe in enumerate(sub_graph):
            #     pair, score = dupe
            #     costs.append(1-score)

            # edges = numpy.array(edges)
            # costs = numpy.array(costs)

            # segs = segment(edges, costs, scale_param, smoothing_param)


            # ####################################
            # ########### igraph #################
            # ####################################
            candidate_set = numpy.unique(sub_graph['pairs'])
            candidate_set = numpy.sort(candidate_set)

            (i_to_id, condensed_distances) = condensedDistance(sub_graph)
            distances = hcluster.squareform(condensed_distances)

            g = Graph()
            g.add_vertices(len(candidate_set))
            edges = candidate_set.searchsorted(sub_graph['pairs'])
            g.add_edges(edges)

            edge_weights = []
            for pair, score in sub_graph:
                edge_weights.append(score)

            # # 1. community_fastgreedy - automatically selects a # of clusters (that maximizes modularity score)
            # dend = g.community_fastgreedy(edge_weights)
            # clusters = dend.as_clustering()
            # segs = clusters.membership

            # # 2. community_infomap
            # clusters = g.community_infomap(edge_weights)
            # segs = clusters.membership

            # # 3. community_leading_eigenvector
            # # how to set param arpack_options?
            # # this has poor recall w/ default arpack_options = None
            # clusters = g.community_leading_eigenvector(weights = edge_weights)
            # segs = clusters.membership

            # # 4. community_label_propagation
            # clusters = g.community_label_propagation(weights=edge_weights)
            # segs = clusters.membership

            # # 5. community_multilevel
            # # this has poor recall
            # clusters = g.community_multilevel(weights=edge_weights)
            # segs = clusters.membership

            # # 6. community_edge_betweenness
            # dend = g.community_edge_betweenness(directed=False, weights=edge_weights)
            # clusters = dend.as_clustering()
            # segs = clusters.membership

            # 7. community_spinglass? what is the spinglass community detection method?

            # 8. community_walktrap
            # really poor recall. perhaps set param steps?
            # steps = length of random walks, default 4 (4 seems long considering the size of many subgraphs)
            dend = g.community_walktrap(weights=edge_weights)
            clusters = dend.as_clustering()
            segs = clusters.membership


            num_clusters = len(set(segs))
            for i in range(num_clusters):
                cluster_members = [idx for idx in range(len(candidate_set)) if segs[idx] == i]
                if len(cluster_members) > 1 :
                    scores = confidences(cluster_members, distances)
                    all_clusters.append( (tuple(i_to_id[item] for item in cluster_members), tuple(scores)) )
        #only one pair in sub graph
        else:
            ids, score = sub_graph[0]
            all_clusters.append( (tuple(ids), tuple([score]*2)) )

    return all_clusters

def confidences(items, distances) :
    scores = numpy.sum(distances[items, :][:, items]**2, 0)
    scores /= len(items) - 1
    scores = 1 - numpy.sqrt(scores)
    return scores

def greedyMatching(dupes, threshold=0.5):
    dupes = numpy.array(dupes)
    covered_vertex_A = set([])
    covered_vertex_B = set([])
    clusters = []

    sorted_dupes = sorted(dupes, key=lambda score: score[1], reverse=True)
    dupes_list = [dupe for dupe in sorted_dupes if dupe[1] >= threshold]

    for vertices, score in dupes_list:
        a, b = vertices
        if a not in covered_vertex_A and b not in covered_vertex_B:
            clusters.append((vertices, score))
            covered_vertex_A.add(a)
            covered_vertex_B.add(b)

    return clusters

def gazetteMatching(dupes, threshold=0.5, n=1):
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
                if i < n :
                    matches.append((pair, score))
                    i += 1
            else :
                clusters.append(tuple(matches))
                matches = [(pair, score)]
                i = 1
                group = a

        clusters.append(tuple(matches))

    return clusters
