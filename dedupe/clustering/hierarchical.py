import numpy
import fastcluster
import hcluster

def condensedDistance(dupes) :
  # Convert the pairwise list of distances in dupes to "condensed
  # distance matrix" required by the hierarchical clustering
  # algorithms. Also return a dictionary that maps the distance matrix
  # to the record_ids.
  #
  # The condensed distance matrix is described in the scipy
  # documentation of scipy.cluster.hierarchy.linkage
  # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

  
  candidate_set = set([])
  for pair, _ in dupes :
    candidate_set.update(pair)

  remap = dict([(candidate_id, i) for i, candidate_id
                in enumerate(sorted(list(candidate_set)))])

  N = len(remap)
  matrix_length = (N * (N-1))/2 

  condensed_distances = [1] * matrix_length

  for pair, score in dupes :
    (i, j) = (remap[pair[0]], remap[pair[1]])
    if i > j :
      i,j = j,i
    subN = ((N - i)*(N - i - 1))/2
    index = matrix_length - subN + j - i - 1
    condensed_distances[index] = 1-score

  return remap, condensed_distances

def cluster(dupes, threshold) :
  remap, condensed_distances = condensedDistance(dupes) 
  linkage = fastcluster.linkage(numpy.array(condensed_distances),
                          method='centroid')
  partition = hcluster.fcluster(linkage, threshold)

  clustering = {}
  
  for cluster_id, record_id in zip(partition, remap.keys()) :
    clustering.setdefault(cluster_id, []).append(record_id)

  clusters = [set(l) for l in clustering.values() if len(l) > 1]

  return clusters

