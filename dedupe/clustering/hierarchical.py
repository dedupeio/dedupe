#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import fastcluster
import hcluster

def condensedDistance(dupes):

   # Convert the pairwise list of distances in dupes to "condensed
   # distance matrix" required by the hierarchical clustering
   # algorithms. Also return a dictionary that maps the distance matrix
   # to the record_ids.
   #
   # The condensed distance matrix is described in the scipy
   # documentation of scipy.cluster.hierarchy.linkage
   # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    candidate_set = numpy.unique(dupes['pairs'])
    candidate_set = numpy.sort(candidate_set)
    
    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes['pairs'])
    id_1 = ids[:,0]
    id_2 = ids[:,1]

    N = len(candidate_set)
    matrix_length = N * (N - 1) / 2

    step = (N - id_1) * (N - id_1 - 1) / 2
    index = matrix_length - step + id_2 - id_1 -1

    condensed_distances = numpy.empty(matrix_length, 'f4')
    condensed_distances[index] = 1 - dupes['score']

    return (i_to_id, condensed_distances)

def cluster(dupes, threshold=.5):
    """
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    Keyword arguments:
    threshold -- number betweent 0 and 1 (default is .5). lowering the 
                 number will increase precision, raising it will increase
                 recall
    """
    (i_to_id, condensed_distances) = condensedDistance(dupes)
    linkage = fastcluster.linkage(condensed_distances,
                                  method='centroid',
                                  preserve_input=False)

    partition = hcluster.fcluster(linkage, threshold)

    clustering = {}

    for (i, cluster_id) in enumerate(partition):
        clustering.setdefault(cluster_id, []).append(i_to_id[i])

    clusters = [set(l) for l in clustering.values() if len(l) > 1]

    return clusters
