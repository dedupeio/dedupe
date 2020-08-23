#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from collections import defaultdict
import array
import logging
from typing import (Iterable,
                    Dict,
                    ValuesView,
                    cast,
                    List,
                    Set,
                    Generator,
                    Sequence,
                    Tuple)
from dedupe._typing import Clusters, RecordID, Links
import numpy
import fastcluster
import hcluster

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def connected_components(edgelist: numpy.ndarray,
                         max_components: int) -> Generator[numpy.ndarray, None, None]:

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


def union_find(edgelist: numpy.ndarray) -> ValuesView[Sequence[int]]:

    root: Dict[RecordID, RecordID] = {}
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
            root_a = cast(RecordID, root_a)  # AH upgrade
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


def condensed_distance(dupes: numpy.ndarray) -> Tuple[Dict[int, RecordID],
                                                      numpy.ndarray,
                                                      int]:
    """Convert the pairwise list of distances in dupes to ``condensed distance vector``.

    The condensed distance vector is required by the hierarchical clustering
    algorithms: http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    Let's suppose we have 3 records. Then there are 3_C_2 = 3 possible pairs. Our
    distance matrix might look like this:

    ::
              a        b       c
        a  0        d(a, b)  d(a, c)
        b  d(b, a)  0        d(b, c)
        c  d(c, a)  d(c, b)  0

    Since that contains some redundant information, we create a condensed distance vector
    from the upper right triangular of the distance matrix. We just read from
    left to right.

    ::

        [d(a, b), d(a, c), d(b, c)]

    The formula for an index of the condensed matrix is

    index = {N choose 2}-{N-row choose 2} + (col-row-1)
          = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1
            ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
          matrix_length       row_step

    where (row,col) is index of an uncondensed square N X N distance matrix.

    Returns:
        i_to_id: (dict) dictionary that maps the distance matrix to the record_ids.
        condensed_distances: (np.array) a 1 x N_C_2 dimensional vector, containing all the
            pair-wise distances flattened into a 1D array.
        N: (int)

    """

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


def cluster(dupes: numpy.ndarray,
            cluster_threshold: float = 0.5,
            max_components: int = 30000,
            id_to_match: str = None) -> Clusters:
    """
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    `https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.fcluster.html`



    Args:
        dupes: (np.array)[tuple(list[str], float)] A list of tuples, where each tuple
            contains an id pair and a probability that they are a match:
                id_pair_tuple: ([record_id_1, record_id_2], prob)
                dtype: np.dtype([('pairs', '<U256', 2),
                                 ('score', 'f4', 1)])
        threshold: (float) number betweent 0 and 1 (default is .5). lowering the
            number will increase precision, raising it will increase recall
    """
    distance_threshold = cluster_threshold
    score_threshold = 1 - cluster_threshold
    dupe_sub_graphs = connected_components(dupes, max_components)
    # logger.info(f"Dupes: {dupes}")
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:
            i_to_id, condensed_distances, N = condensed_distance(sub_graph)
            logger.debug(f"{condensed_distances}")
            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid',
                                          preserve_input=True)
            partition = hcluster.fcluster(linkage,
                                          distance_threshold,
                                          criterion='distance')

            clusters: Dict[int, List[int]] = defaultdict(list)
            logger.debug(f"Partition: {partition}")
            logger.debug(f"Linkage: {linkage}")
            for i, cluster_id in enumerate(partition):
                clusters[cluster_id].append(i)

            logger.info(f"Clusters: {clusters}")
            for cluster in clusters.values():
                if len(cluster) > 1:
                    scores = confidences(cluster, condensed_distances, N)
                    logger.info(f"{tuple(i_to_id[i] for i in cluster)}, {scores}")
                    ids = [i_to_id[i] for i in cluster]
                    if id_to_match in ids and id_to_match is not None:
                        yield tuple(ids), scores
                    elif id_to_match is None:
                        yield tuple(ids), scores
                    # yield tuple(i_to_id[i] for i in cluster), scores

        else:
            (ids, score), = sub_graph
            if score > score_threshold and id_to_match in ids and id_to_match is not None:
                # logger.info(tuple(ids), ((score,) * 2))
                yield tuple(ids), (score,) * 2
            elif score > score_threshold and id_to_match is None:
                yield tuple(ids), (score,) * 2


def confidences(cluster: Sequence[int],
                condensed_distances: numpy.ndarray,
                d: int) -> numpy.ndarray:
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


def greedyMatching(dupes: numpy.ndarray, threshold: float = 0.5) -> Links:  # AH upgrade threshold
    A: Set[RecordID] = set()
    B: Set[RecordID] = set()

    dupes = dupes[dupes['score'] >= threshold]
    dupes.sort(order='score')
    dupes = dupes[::-1]

    for (a, b), score in dupes:
        if a not in A and b not in B:
            A.add(a)
            B.add(b)

            yield (a, b), score


def gazetteMatching(scored_blocks: Iterable[numpy.ndarray],
                    threshold: float = 0,
                    n_matches: int = 1) -> Links:  # AH upgrade threshold

    for block in scored_blocks:
        block.sort(order='score')
        block = block[::-1]

        if n_matches:
            yield block[:n_matches].copy()
        else:
            yield block.copy()


def pair_gazette_matching(scored_pairs: numpy.ndarray,
                          threshold: float = 0.0,
                          n_matches: int = 1) -> Links:

    scored_pairs.sort(order='pairs')

    group_key = scored_pairs['pairs'][:, 0]
    change_points = numpy.where(numpy.roll(group_key, 1) != group_key)[0]
    scored_blocks = numpy.split(scored_pairs, change_points)

    for match in gazetteMatching(scored_blocks, threshold, n_matches):
        if match:
            yield from match
