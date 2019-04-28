#!/usr/bin/python
# -*- coding: utf-8 -*-
from future.utils import viewvalues

import itertools
from collections import defaultdict
import array
import logging

import numpy
import fastcluster
import hcluster

logger = logging.getLogger(__name__)


def connected_components(edgelist, max_components):

    if len(edgelist) == 0:
        raise StopIteration()

    components = union_find(edgelist['pairs'])

    for component in components:
        sub_graph = edgelist[component]
        n_components = len(numpy.unique(sub_graph['pairs']))

        if n_components > max_components:
            min_score = numpy.min(sub_graph['score'])
            min_score_logit = numpy.log(min_score) - numpy.log(1 - min_score)
            threshold = 1 / (1 + numpy.exp(-min_score_logit - 1))
            logger.warning('A component contained %s elements. '
                           'Components larger than %s are '
                           're-filtered. The threshold for this '
                           'filtering is %s' % (n_components,
                                                max_components,
                                                threshold))
            filtered_sub_graph = sub_graph[sub_graph['score'] > threshold]
            for sub_graph in connected_components(filtered_sub_graph,
                                                  max_components):
                yield sub_graph
        else:
            yield sub_graph


def union_find(edgelist):

    root = {}
    components = {}
    component_size = {}

    it = numpy.nditer(edgelist, ['external_loop'])

    for i, (a, b) in enumerate(it):
        root_a = root.get(a)
        root_b = root.get(b)

        if root_a is None and root_b is None:
            # assuming that it will be a while before we are handling
            # edgelists of much more than 4 billion elements we will
            # use an the 'I' type
            components[a] = array.array('I', [i])
            component_size[a] = 2
            root[a] = root[b] = a
        elif root_a is None or root_b is None:
            if root_a is None:
                b = a
                root_a = root_b
            components[root_a].append(i)
            component_size[root_a] += 1
            root[b] = root_a
        elif root_a != root_b:
            if component_size[root_a] < component_size[root_b]:
                root_a, root_b = root_b, root_a

            components[root_a].extend(components[root_b])
            components[root_a].append(i)

            component_b = numpy.unique(edgelist[components[root_b]])

            for node in component_b:
                root[node] = root_a

            component_size[root_a] += len(component_b)

            del components[root_b]
            del component_size[root_b]

        else:
            components[root_a].append(i)

    return components.values()


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

    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes['pairs'])
    row = ids[:, 0]
    col = ids[:, 1]

    N = len(candidate_set)
    matrix_length = N * (N - 1) / 2

    row_step = (N - row) * (N - row - 1) / 2
    index = matrix_length - row_step + col - row - 1

    condensed_distances = numpy.ones(int(matrix_length), 'f4')
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
    distance_threshold = 1 - threshold
    dupe_sub_graphs = connected_components(dupes, max_components)

    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:

            i_to_id, condensed_distances, N = condensedDistance(sub_graph)

            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid',
                                          preserve_input=True)

            partition = hcluster.fcluster(linkage,
                                          distance_threshold,
                                          criterion='distance')

            clusters = defaultdict(list)

            for i, cluster_id in enumerate(partition):
                clusters[cluster_id].append(i)

            for cluster in viewvalues(clusters):
                if len(cluster) > 1:
                    scores = confidences(cluster, condensed_distances, N)
                    yield tuple(i_to_id[i] for i in cluster), scores

        else:
            (ids, score), = sub_graph
            if score > threshold:
                yield tuple(ids), (score,) * 2


def confidences(cluster, condensed_distances, d):
    '''
    We calculate a per record score that is similar to a standard
    deviation.  The main reason is that these record scores can be
    used to calculate the standard deviation of an entire cluster,
    which is a reasonable metric for clusters.
    '''

    scores = dict.fromkeys(cluster, 0.0)
    squared_distances = condensed_distances ** 2
    for i, j in itertools.combinations(cluster, 2):
        index = d * (d - 1) / 2 - (d - i) * (d - i - 1) / 2 + j - i - 1
        squared_dist = squared_distances[int(index)]
        scores[i] += squared_dist
        scores[j] += squared_dist
    scores = numpy.array([score for _, score in sorted(scores.items())])
    scores /= len(cluster) - 1
    scores = numpy.sqrt(scores)
    scores = 1 - scores
    return scores


def greedyMatching(dupes, threshold=0.5):
    A = set()
    B = set()

    dupes = dupes[dupes['score'] >= threshold]
    dupes.sort(order='score')
    dupes = dupes[::-1]

    for (a, b), score in dupes:
        if a not in A and b not in B:
            A.add(a)
            B.add(b)

            yield (a, b), score


def gazetteMatching(scored_blocks, n_matches=1):

    for block in scored_blocks:
        block.sort(order='score')
        block = block[::-1]

        if n_matches:
            yield block[:n_matches]
        else:
            yield block
