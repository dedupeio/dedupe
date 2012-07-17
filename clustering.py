from collections import defaultdict
from itertools import combinations

def memoize(f):
  cache= {}
  def memf(*x):
    x_key = (tuple(x[0]), x[1])
    if x_key not in cache:
      cache[x_key] = f(*x)
    return cache[x_key]

  return memf

def neighborDict(duplicates) :
  neighbors = defaultdict(list)

  for pair, similarity in duplicates :
    candidate_1, candidate_2 = pair
    distance = 1-similarity

    neighbors[candidate_1].append((candidate_2, distance))
    neighbors[candidate_2].append((candidate_1, distance))
    
  for candidate in neighbors :
    neighbors[candidate] = sorted(neighbors[candidate],
                                  key = lambda neighborhood : neighborhood[1])
                                   
  return neighbors

@memoize
def neighborhoodGrowth(neighborhood, neighborhood_multiplier) :
  distances = zip(*neighborhood)[1]
  smallest_distance = min(distances)
  neighborhood_growth = sum([distance < (neighborhood_multiplier
                                         * smallest_distance)
                             for distance in distances])

  return neighborhood_growth

def kOverlap(neighborhood_1, neighborhood_2) :
  K = min(len(neighborhood_1), len(neighborhood_2))
  overlap = [False] * K

  if set(neighborhood_1[:K+1]).intersection(set(neighborhood_2[:K+1])) :
    for k in range(1,K+1) :
      if set(neighborhood_1[:k]) == set(neighborhood_2[:k]) :
        overlap[k-1] = True

  return overlap
  
def compactPairs(neighbors,
                 neighborhood_multiplier,
                 k_nearest_neighbors,
                 sparseness_threshold) :
  compact_pairs = []

  candidates = neighbors.keys()
  candidates = sorted(candidates)
  candidate_pairs = combinations(candidates, 2)

  for pair in candidate_pairs :
    candidate_1, candidate_2 = pair

    # This is appropriate if the aggregate function for the Spatial
    # Neighborhood Threshold is MAX, not if its AVG
    neighbors_1 = neighbors[candidate_1]
    ng_1 = neighborhoodGrowth(neighbors_1, neighborhood_multiplier) 
    if ng_1 > sparseness_threshold :
      continue
      
    neighbors_2 = neighbors[candidate_2]
    ng_2 = neighborhoodGrowth(neighbors_2, neighborhood_multiplier) 
    if ng_2 > sparseness_threshold :
      continue

    # Include candidates in list of neighbors of candidate so
    # that 1 : [2, 3] and 2 : [1,3] will become identical sets
    # 1 : [1, 2, 3] and 2 : [2, 1, 3]  
    
    neighb_candidates_1 = list(zip(*neighbors_1)[0][:k_nearest_neighbors])
    neighb_candidates_1.insert(0, candidate_1)

    neighb_candidates_2 = list(zip(*neighbors_2)[0][:k_nearest_neighbors])
    neighb_candidates_2.insert(0, candidate_2)

    k_set_overlap = kOverlap(neighb_candidates_1,
                             neighb_candidates_2)


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
<<<<<<< HEAD
  cluster = set([])
  assigned_candidates = set([])
  
  for pair in compact_pairs :
    candidate_1, candidate_2 = pair
    if candidate_2 not in assigned_candidates :
        
      max_growth = max(neighborhood_attributes[candidate_1]['neighborhood growth'],
                       neighborhood_attributes[candidate_2]['neighborhood growth'])
      if max_growth <= sparseness_threshold :
        if candidate_1 in cluster :
          cluster.add(candidate_2)
          assigned_candidates.add(candidate_2)
        elif cluster :
          clusters.append(cluster)
          cluster = set(pair)
          assigned_candidates.update(pair)
        else :
          cluster = set(pair)

    
  if cluster :
    clusters.append(cluster)
=======

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
>>>>>>> upstream/master

  #print assigned_candidates
  return clusters

  
def growthDistributions(neighbors, neighborhood_multiplier) :
  growths = []

  for neighborhood in neighbors.values() :
    growths.append(neighborhoodGrowth(neighborhood, neighborhood_multiplier))

  distribution = [(growths.count(growth),
                   growth) 
                  for growth in set(growths)]
                    
  distribution = sorted(distribution, key = lambda growth : growth[1])
  
<<<<<<< HEAD
  #print "ng_distribution"                  
  #print ng_distribution
=======
  distribution = [(growth[0]/float(len(growths)),
                   growth[1])
                  for growth in distribution]

>>>>>>> upstream/master
  
  cumulative_distribution = []
  cumulative_growth = 0
  for i, growth in enumerate(distribution) :
    cumulative_growth += growth[0]
    cumulative_distribution.append((cumulative_growth, growth[1]))
       
<<<<<<< HEAD
  #print "ng_cumulative_distribution"                  
  #print ng_cumulative_distribution
  
  return ng_distribution, ng_cumulative_distribution
=======
  return distribution, cumulative_distribution
>>>>>>> upstream/master
  
def sparsenessThreshold(neighbors,
                        estimated_dupe_fraction,
                        epsilon = 0.05,
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

  if (len(fraction_window) == 0) :
    return distribution[i][1]
  
  else :
    # of the quantiles found, return minimum spike
    for j in range(1, len(fraction_window)) :
      if (distribution[fraction_window[j]][0]
          - distribution[fraction_window[j-1]][0]) > 0 :
        return distribution[fraction_window[j]][1]
        
  return distribution[fraction_window[-1]][1]


    
def cluster(duplicates,
            sparseness_threshold = 4,
            k_nearest_neighbors = 6,
            neighborhood_multiplier = 2,
            estimated_dupe_fraction = None) :

  neighbors = neighborDict(duplicates)

  if estimated_dupe_fraction :
    sparseness_threshold = sparsenessThreshold(neighbors,
                                               estimated_dupe_fraction)

  compact_pairs = compactPairs(neighbors,
                               neighborhood_multiplier,
                               k_nearest_neighbors,
                               sparseness_threshold)
  return(partition(compact_pairs))





  

