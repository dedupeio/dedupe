#!/usr/bin/python
from __future__ import annotations

import array
import itertools
import logging
import tempfile
from collections import defaultdict
from typing import Generator, Iterable, Sequence, cast

import numpy
import numpy.typing
import scipy.cluster.hierarchy

from dedupe._typing import ArrayLinks, Clusters, RecordID, Scores, TupleLinks

logger = logging.getLogger(__name__)


def connected_components(
    edgelist: Scores, max_components: int
) -> Generator[Scores, None, None]:
    if len(edgelist) == 0:
        raise StopIteration()

    unlabeled_edgelist = edgelist

    # we are going to keep track of the connected components
    # with another field in the record array of the edgelist.
    # unfortunately, it's not straightforward to add another
    # field to a memmapped record array so, we are going to
    # have to create a new memmapped array with all the fields
    # we want and copy things over.
    with tempfile.TemporaryDirectory() as path:
        filename = path + "/unlabeled_edgelist"
        edgelist = numpy.memmap(
            filename,
            dtype=(unlabeled_edgelist.dtype.descr + [("label", "int32")]),
            mode="w+",
            shape=unlabeled_edgelist.shape,
        )

        if hasattr(unlabeled_edgelist, "filename"):
            assert isinstance(unlabeled_edgelist, numpy.memmap)
            copy_mmap_record_arrays(unlabeled_edgelist, edgelist, ["pairs", "score"])
        else:
            copy_to_mmap_record_array(unlabeled_edgelist, edgelist, ["pairs", "score"])

        yield from _connected_components(edgelist, max_components)

        edgelist._mmap.close()  # type: ignore


def _connected_components(
    edgelist: Scores, max_components: int
) -> Generator[Scores, None, None]:
    component_stops = union_find(edgelist)

    start = 0
    for stop in component_stops:
        sub_graph = edgelist[start:stop]
        n_edges = stop - start
        start = stop

        needs_filtering = False
        # first we find the upper bound of the
        # number of components given the edgelist
        upper_bound_components = n_edges + 1
        if upper_bound_components > max_components:
            # which we can refine using a more expensive operation
            # if it's possible we have too many components
            n_components = len(numpy.unique(sub_graph["pairs"]))
            if n_components > max_components:
                needs_filtering = True

        if needs_filtering:
            min_score = numpy.min(sub_graph["score"])
            min_score_logit = numpy.log(min_score) - numpy.log(1 - min_score)
            threshold = 1 / (1 + numpy.exp(-min_score_logit - 1))
            logger.warning(
                "A component contained %s elements. "
                "Components larger than %s are "
                "re-filtered. The threshold for this "
                "filtering is %s" % (n_components, max_components, threshold)
            )
            # slices of memmaped arrays are also memmaped arrays,
            # which is what we want. The components should
            # already sorted by score so we can slice as oppose
            # to selecting like `sub_graph[sub_graph['score'] >
            # threshold]`, which would lead to an in memory copy being
            # made
            cut_point = numpy.searchsorted(sub_graph["score"], threshold)
            filtered_sub_graph = sub_graph[max(cut_point, 2) :]

            for sub_graph in _connected_components(filtered_sub_graph, max_components):
                yield sub_graph[["pairs", "score"]]
        else:
            yield sub_graph[["pairs", "score"]]


def union_find(scored_pairs: Scores) -> numpy.typing.NDArray[numpy.int_]:
    root: dict[RecordID, int] = {}

    components = {}

    edgelist = scored_pairs["pairs"]
    labels = scored_pairs["label"]

    it = numpy.nditer(edgelist, ["external_loop"])

    n_edges = len(scored_pairs)

    array_type = "H"
    for array_type in ("H", "I", "L", "Q"):
        if (2 ** (8 * array.array(array_type).itemsize) - 1) > n_edges:
            break
    else:
        raise ValueError(
            "You are making more comparisons than your machine can handle, try raising the threshold"
        )

    a: RecordID
    b: RecordID

    for i, (a, b) in enumerate(it):  # type: ignore
        root_a = root.get(a)
        root_b = root.get(b)

        if root_a is None and root_b is None:
            root[a] = root[b] = i
            components[i] = array.array(array_type, [i])
        elif root_a is None or root_b is None:
            if root_a is None:
                b = a
                root_a = root_b
            root_a = cast(int, root_a)
            components[root_a].append(i)
            root[b] = root_a
        elif root_a != root_b:
            if len(components[root_a]) < len(components[root_b]):
                root_a, root_b = root_b, root_a

            components[root_a].extend(components[root_b])
            components[root_a].append(i)

            component_b = numpy.unique(edgelist[components[root_b]])
            for node in component_b:
                root[node] = root_a

            del components[root_b]

        else:
            components[root_a].append(i)

    for label, component in components.items():
        labels[component] = label

    # we want our selections to remain memmapped arrays so we sort and
    # get the indices where the components change. This will allow us
    # to slice pieces of the memmapped array, and have those slices
    # also be memmaped arrays.
    #
    # Since we we might be calling union_find recursively it would be
    # good to use a stable sort to take advantage of sorting done in a
    # previous recursion.
    #
    # stable sorts in numpy take about n / 2 work space so we won't
    # do it if scored_pairs is too big
    if len(scored_pairs) > 2500000:
        scored_pairs.sort(order=("label", "score"))
    else:
        scored_pairs.sort(order=("label", "score"), kind="stable")

    return numpy.cumsum(numpy.unique(scored_pairs["label"], return_counts=True)[1])


def condensedDistance(
    dupes: Scores,
) -> tuple[dict[int, RecordID], numpy.typing.NDArray[numpy.float64], int]:
    """
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
    """

    candidate_set = numpy.unique(dupes["pairs"])

    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes["pairs"])
    row = ids[:, 0]
    col = ids[:, 1]

    N = len(candidate_set)

    # alternate form thanks to wolfram alpa
    index = row * (2 * N - row - 3) // 2 + col - 1

    condensed_distances = numpy.ones(N * (N - 1) // 2, "f4")
    condensed_distances[index] = 1 - dupes["score"]

    return i_to_id, condensed_distances, N


def cluster(
    dupes: Scores, threshold: float = 0.5, max_components: int = 30000
) -> Clusters:
    """
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    Keyword arguments:
    threshold -- number between 0 and 1 (default is .5). lowering the
                 number will increase precision, raising it will increase
                 recall
    """
    distance_threshold = 1 - threshold
    dupe_sub_graphs = connected_components(dupes, max_components)

    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:
            i_to_id, condensed_distances, N = condensedDistance(sub_graph)

            linkage = scipy.cluster.hierarchy.linkage(
                condensed_distances, method="centroid"
            )

            partition = scipy.cluster.hierarchy.fcluster(
                linkage, distance_threshold, criterion="distance"
            )

            clusters: dict[int, list[int]] = defaultdict(list)

            for i, cluster_id in enumerate(partition):
                clusters[cluster_id].append(i)

            squared_distances = condensed_distances**2
            for cluster in clusters.values():
                if len(cluster) > 1:
                    scores = confidences(cluster, squared_distances, N)
                    yield tuple(i_to_id[i] for i in cluster), scores  # type: ignore[misc]

        else:
            ((ids, score),) = sub_graph
            if score > threshold:
                yield tuple(ids), (score,) * 2


def confidences(
    cluster: Sequence[int],
    squared_distances: numpy.typing.NDArray[numpy.float64],
    d: int,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    We calculate a per record score that is similar to a standard
    deviation.  The main reason is that these record scores can be
    used to calculate the standard deviation of an entire cluster,
    which is a reasonable metric for clusters.
    """
    scores: numpy.typing.NDArray[numpy.float64]
    scores_d = dict.fromkeys(cluster, 0.0)
    C = 2 * d - 3
    for i, j in itertools.combinations(cluster, 2):
        index = i * (C - i) // 2 + j - 1
        squared_dist = squared_distances[index]
        scores_d[i] += squared_dist
        scores_d[j] += squared_dist
    scores = numpy.array([score for _, score in sorted(scores_d.items())])
    scores /= len(cluster) - 1
    scores = numpy.sqrt(scores)
    scores = 1 - scores
    return scores


def greedyMatching(dupes: Scores) -> TupleLinks:
    A: set[RecordID] = set()
    B: set[RecordID] = set()

    dupes.sort(order="score")
    dupes = dupes[::-1]

    for (a, b), score in dupes:
        if a not in A and b not in B:
            A.add(a)
            B.add(b)

            yield (a, b), score


def gazetteMatching(
    scored_blocks: Iterable[Scores], threshold: float = 0, n_matches: int = 1
) -> ArrayLinks:
    for block in scored_blocks:
        block = block[block["score"] > threshold]
        block.sort(order="score")
        block = block[::-1]

        if len(block):
            if n_matches:
                yield block[:n_matches].copy()
            else:
                yield block.copy()


def pair_gazette_matching(
    scored_pairs: Scores, threshold: float = 0.0, n_matches: int = 1
) -> ArrayLinks:
    scored_pairs.sort(order="pairs")

    group_key = scored_pairs["pairs"][:, 0]
    change_points = numpy.where(numpy.roll(group_key, 1) != group_key)[0]
    scored_blocks = numpy.split(scored_pairs, change_points)

    for match in gazetteMatching(scored_blocks, threshold, n_matches):
        if match:
            yield from match


def copy_to_mmap_record_array(
    source: numpy.ndarray,
    target: numpy.memmap,
    fields: list[str],
    chunksize: int = 100000,
) -> None:
    """
    Writing into a memmapped array allocates memory equivalent to the
    amount that you are writing. With big arrays this is undesirable
    so we write in chunks
    """
    assert target.filename is not None

    start = 0
    stops = itertools.chain(range(chunksize, source.size, chunksize), [source.size])
    for stop in stops:
        shape = (stop - start,)
        source_slice = source[start:stop]
        target_slice: numpy.memmap = numpy.memmap(
            target.filename,
            dtype=target.dtype,
            offset=(start * target.dtype.itemsize),
            shape=shape,
        )
        target_slice[fields] = source_slice[fields]
        start = stop


def copy_mmap_record_arrays(
    source: numpy.memmap,
    target: numpy.memmap,
    fields: list[str],
    chunksize: int = 100000,
) -> None:
    """
    Writing into a memmapped array allocates memory equivalent to the
    amount that you are writing. With big arrays this is undesirable
    so we write in chunks
    """
    assert source.filename is not None
    assert target.filename is not None

    start = 0
    stops = itertools.chain(range(chunksize, source.size, chunksize), [source.size])
    for stop in stops:
        shape = (stop - start,)
        source_slice: numpy.memmap = numpy.memmap(
            source.filename,
            dtype=source.dtype,
            offset=(start * source.dtype.itemsize),
            shape=shape,
        )
        target_slice: numpy.memmap = numpy.memmap(
            target.filename,
            dtype=target.dtype,
            offset=(start * target.dtype.itemsize),
            shape=shape,
        )
        target_slice[fields] = source_slice[fields]

        start = stop
