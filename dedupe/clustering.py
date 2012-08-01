from collections import defaultdict
from itertools import combinations

def neighborDict(duplicates) :
  neighbors = defaultdict(list)

  for pair, similarity in duplicates :
    candidate_1, candidate_2 = pair
    distance = 1 - similarity

    neighbors[candidate_1].append((candidate_2, distance))
    neighbors[candidate_2].append((candidate_1, distance))
    
  for candidate in neighbors :
    neighbors[candidate] = sorted(neighbors[candidate],
                                  key = lambda neighborhood : neighborhood[1])
                                   
  return neighbors

def neighborhoodGrowth(neighborhood, neighborhood_multiplier) :
  distances = zip(*neighborhood)[1]
  smallest_distance = min(distances)
  neighborhood_growth = sum([distance <= (neighborhood_multiplier
                                          * smallest_distance)
                             for distance in distances])

  return neighborhood_growth

def kOverlap(neighborhood_1, neighborhood_2) :
  K = min(len(neighborhood_1), len(neighborhood_2))
  overlap = [False] * K

  for k in xrange(1,K+1) :
    if set(neighborhood_1[:k]) == set(neighborhood_2[:k]) :
      overlap[k-1] = True

  return overlap

def compactPairs(neighbors) :

  compact_pairs = []

  candidates = neighbors.keys()
  candidates = sorted(candidates)
  candidate_pairs = combinations(candidates, 2)

  for pair in candidate_pairs :
    candidate_1, candidate_2 = pair

    # This is appropriate if the aggregate function for the Spatial
    # Neighborhood Threshold is MAX, not if its AVG
    neighbors_1 = neighbors[candidate_1]
      
    neighbors_2 = neighbors[candidate_2]

    # Include candidates in list of neighbors of candidate so
    # that 1 : [2, 3] and 2 : [1,3] will become identical sets
    # 1 : [1, 2, 3] and 2 : [2, 1, 3]  
    
    if (candidate_1 in neighbors_2
        and candidate_2 in neighbors_1) :
      k_set_overlap = kOverlap(neighbors_1,
                               neighbors_2)


      if any(k_set_overlap) :
        # Since the nearest neighbor to a candidate is always itself the
        # first elements will never overlap
        k_set_overlap = k_set_overlap[1:]

        compact_pairs.append((pair,
                              k_set_overlap))

    
  return compact_pairs

def partition(compact_pairs) :

  assigned_candidates = set([])
  clusters = []

  groups = defaultdict(list)
  for pair in compact_pairs :
    groups[pair[0][0]].append(pair)

  for group_id, group in groups.iteritems() :
    if group_id not in assigned_candidates :
      pair_ids, k_compact_set = zip(*group)

      compact_set_sizes = [sum(compact_bool) for compact_bool
                           in zip(*k_compact_set)]

      k = compact_set_sizes.index(max(compact_set_sizes))

      cluster = set([])
      for i, compact_bool in enumerate(k_compact_set) :
        if compact_bool[k] :
          cluster.update(pair_ids[i])
          assigned_candidates.update(pair_ids[i])

      if cluster :
        clusters.append(cluster)

  return clusters

  
def growthDistributions(neighbors, neighborhood_multiplier) :
  growths = []

  for neighborhood in neighbors.values() :
    growths.append(neighborhoodGrowth(neighborhood, neighborhood_multiplier))

  distribution = [(growths.count(growth),
                   growth) 
                  for growth in set(growths)]
                    
  distribution = sorted(distribution, key = lambda growth : growth[1])
  
  distribution = [(growth[0]/float(len(growths)),
                   growth[1])
                  for growth in distribution]

  
  cumulative_distribution = []
  cumulative_growth = 0
  for i, growth in enumerate(distribution) :
    cumulative_growth += growth[0]
    cumulative_distribution.append((cumulative_growth, growth[1]))

  print "Distribution of Growths"
  for quantile in distribution :
    print ("%.2f" % quantile[0], quantile[1])
  
       
  return distribution, cumulative_distribution

def sparsenessThreshold(neighbors,
                        estimated_dupe_fraction,
                        epsilon = 0.1,
                        neighborhood_multiplier=2) :

  (distribution,
   cumulative_distribution) = growthDistributions(neighbors, 2)

  growth_quantiles = zip(*cumulative_distribution)[0]

  fraction_window = []
  for i, quantile in enumerate(growth_quantiles) :
    if quantile > (estimated_dupe_fraction + epsilon) :
      break
    elif quantile > (estimated_dupe_fraction - epsilon) :
      fraction_window.append(i)



  # of the quantiles found, return minimum spike
  for j in range(1, len(fraction_window)) :
    if (distribution[fraction_window[j]][0]
        - distribution[fraction_window[j-1]][0]) > 0 :
      return distribution[fraction_window[j]][1]
        
  return distribution[i+1][1]


    
def cluster(duplicates,
            sparseness_threshold = 4,
            k_nearest_neighbors = 6,
            neighborhood_multiplier = 2,
            estimated_dupe_fraction = None) :

  neighbors = neighborDict(duplicates)

  if estimated_dupe_fraction :
    sparseness_threshold = sparsenessThreshold(neighbors,
                                               estimated_dupe_fraction)
    print "Sparseness Threshold is ", sparseness_threshold

  neighbors = dict([(k, (k,) + zip(*v[:k_nearest_neighbors])[0])
                    for k, v in neighbors.iteritems()
                    if neighborhoodGrowth(v, 2) < sparseness_threshold])

  compact_pairs = compactPairs(neighbors)

  return(partition(compact_pairs))




