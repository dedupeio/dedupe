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

def neighbors(duplicates) :
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
def neighborhoodGrowth(neighborhood, p) :
  distances = zip(*neighborhood)[1]
  smallest_distance = min(distances)
  neighborhood_growth = sum([distance < (p * smallest_distance)
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
  
def compactPairs(neighbors, p, k, sparseness_threshold) :
  compact_pairs = []

  candidates = neighbors.keys()
  candidate_pairs = combinations(candidates, 2)

  for pair in candidate_pairs :
    candidate_1, candidate_2 = pair

    # This is appropriate if the aggregate function for the Spatial
    # Neighborhood Threshold is MAX, not if its AVG
    neighbors_1 = neighbors[candidate_1]
    ng_1 = neighborhoodGrowth(neighbors_1, p) 
    if ng_1 > sparseness_threshold :
      continue
      
    neighbors_2 = neighbors[candidate_2]
    ng_2 = neighborhoodGrowth(neighbors_2, p) 
    if ng_2 > sparseness_threshold :
      continue

    neighb_candidates_1 = list(zip(*neighbors_1)[0][:k])
    neighb_candidates_1.insert(0, candidate_1)

    neighb_candidates_2 = list(zip(*neighbors_2)[0][:k])
    neighb_candidates_2.insert(0, candidate_2)

    k_set_overlap = kOverlap(neighb_candidates_1,
                             neighb_candidates_2)

    k_set_overlap = k_set_overlap[1:]

    if any(k_set_overlap) :
      compact_pairs.append((pair,
                            k_set_overlap,
                            (ng_1, ng_2)))
    
  return compact_pairs

def partition(compact_pairs, sparseness_threshold) :

  assigned_candidates = []

  pairs_ids = compact_pairs.keys()
  pairs_ids = sorted(pairs_ids, key=lambda pair : pair[1])
  pairs_ids = sorted(pairs_ids, key=lambda pair : pair[0])

  for pair in pair_ids :
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

  print assigned_candidates
  return clusters
  
def calculateGrowthDistributions(neighborhood_attributes) :
  neighborhood_growths = []
  for candidate in neighborhood_attributes :
    neighborhood_growths.append(neighborhood_attributes[candidate]['neighborhood growth'])

  ng_distribution = [(neighborhood_growths.count(neighborhood_growth), neighborhood_growth) 
                    for neighborhood_growth in set(neighborhood_growths)]
                    
  ng_distribution = sorted(ng_distribution, key = lambda growth : growth[1])
  
  ng_distribution = [(growth[0]/float(len(neighborhood_growths)), growth[1]) for growth in ng_distribution]
  
  print "ng_distribution"                  
  print ng_distribution
  
  ng_cumulative_distribution = []
  cumulative_growth = 0
  for i, growth in enumerate(ng_distribution) :
    cumulative_growth += growth[0]
    ng_cumulative_distribution.append((cumulative_growth, growth[1]))
       
  print "ng_cumulative_distribution"                  
  print ng_cumulative_distribution
  
  return ng_distribution, ng_cumulative_distribution
  
def calculateSparsenessThreshold(ng_distribution, ng_cumulative_distribution, estimated_dupe_fraction, epsilon = 0.05) :
  fraction_window = []
  i = 0
  growth_quantiles = zip(*ng_cumulative_distribution)[0]
  while growth_quantiles[i] < (estimated_dupe_fraction + epsilon) :
    if (growth_quantiles[i] > (estimated_dupe_fraction - epsilon)) :
      fraction_window.append(i)
    i += 1
      
  print "fraction_window"
  print fraction_window
  
  if (len(fraction_window) == 1) :
    print "fraction_window length is 1, return " , ng_distribution[fraction_window[0]][1]
    return ng_distribution[fraction_window[0]][1]
  
  elif (len(fraction_window) == 0) :
    #return the next largest distribution
    print "fraction_window length is 0, return " , ng_distribution[i][1]
    return ng_distribution[i][1]
  
  else :
    #of the quantiles found, return minimum spike
    print "fraction_window length is > 1"
    for j in range(1, len(fraction_window)) :
      if ((ng_distribution[fraction_window[j]][0] - ng_distribution[fraction_window[j-1]][0]) > 0) :
        return ng_distribution[fraction_window[j]][1]
        
    return ng_distribution[fraction_window[j]][1]
    
def cluster(dupes, threshold, num_nearest_neighbors = 6, neighborhood_multiplier = 2) :
  print "clustering"
  nn = nearestNeighbors(dupes)
  neighborhood_attributes = neighborhoodAttributes(nn, neighborhood_multiplier, num_nearest_neighbors)

  
  print "nearest neighbors"
  print neighborhood_atributes
  
  compact_pairs = compactPairs(neighborhood_attributes)

  print "compact pairs"
  print compact_pairs
  print 'number of compact pairs', len(compact_pairs)
  
  ng_distribution, ng_cumulative_distribution = calculateGrowthDistributions(neighborhood_attributes)
  sparseness_threshold = calculateSparsenessThreshold(ng_distribution, ng_cumulative_distribution, threshold)
  
  print "sparseness_threshold"
  print sparseness_threshold
  
  clustering_partition = partition(compact_pairs, neighborhood_attributes, sparseness_threshold)
  
  print "clustering_partition"
  print clustering_partition
  
  return clustering_partition




  

def neighborhoodAttributes(neighbors, p, K) :
  neighborhood_attributes = {}
  for candidate in nn :
    neighbors = nn[candidate]
    neighbor_ids, proximities = zip(*neighbors)
    closest_proximity = proximities[0]
    neighborhood_growth = sum([proximity < closest_proximity * p for
                               proximity in proximities])
    k = min(K, len(proximities))
    neighborhood_attributes[candidate] = {'neighbor list' : neighbors[0:k],
                                          'neighborhood growth' : neighborhood_growth}

  return neighborhood_attributes
