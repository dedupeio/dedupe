def nearestNeighbors(duplicates) :
  nn = {}
  duplicates = dict(duplicates)
  pairs = duplicates.keys()
  pairs = sorted(pairs, key=lambda pair : pair[1])
  pairs = sorted(pairs, key=lambda pair : pair[0])
  for pair in pairs :
    new_proximity = 1-duplicates[pair]
    candidate_1, candidate_2 = pair
    if candidate_1 in nn :
      for i, neighbor in enumerate(nn[candidate_1]) :
        neighbor_id, proximity = neighbor
        if new_proximity < proximity :
          nn[candidate_1].insert(i, (candidate_2, new_proximity))
          break
      else :
        nn[candidate_1].append((candidate_2, new_proximity))
    else :
      nn[candidate_1] = [(candidate_2, new_proximity)]
    if candidate_2 in nn :
      for i, neighbor in enumerate(nn[candidate_2]) :
        neighbor_id, proximity = neighbor
        if new_proximity < proximity :
          nn[candidate_2].insert(i, (candidate_1, new_proximity))
          break
      else :
        nn[candidate_2].append((candidate_1, new_proximity))
    else :
      nn[candidate_2] = [(candidate_1, new_proximity)]
      
  return nn

def neighborhoodAttributes(nn, p, K) :
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

def compactPairs(neighborhood_attributes) :
  compact_pairs = []
  target_keys = neighborhood_attributes.keys()
  for candidate in neighborhood_attributes :
    candidate_neighbors = neighborhood_attributes[candidate]['neighbor list']
    candidate_neighbor_ids, proximities = zip(*candidate_neighbors)

    target_keys.remove(candidate)

    for target_candidate in target_keys :
      target_neighbors = neighborhood_attributes[target_candidate]['neighbor list']
      target_neighbor_ids, proximities = zip(*target_neighbors)
      if candidate in target_neighbor_ids and target_candidate in candidate_neighbor_ids:
        compact_pairs.append((candidate, target_candidate))

  return compact_pairs

def partition(compact_pairs, neighborhood_attributes, sparseness_threshold) :
  clusters = []
  cluster = set([])
  
  for pair in compact_pairs :
    candidate_1, candidate_2 = pair
    max_growth = max(neighborhood_attributes[candidate_1]['neighborhood growth'],
                     neighborhood_attributes[candidate_2]['neighborhood growth'])
    if max_growth <= sparseness_threshold :
      if cluster.intersection(set(pair)) :
        cluster = cluster.union(set(pair))
      elif cluster :
        clusters.append(cluster)
        cluster = set(pair)
      else :
        cluster = set(pair)

  if cluster :
    clusters.append(cluster)

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
  compact_pairs = compactPairs(neighborhood_attributes)
  
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