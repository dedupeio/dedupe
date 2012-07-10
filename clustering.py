def nearestNeighbors(duplicates) :
  nn = {}
  duplicates = dict(duplicates)
  pairs = duplicates.keys()
  pairs = sorted(pairs, key=lambda pair : pair[1])
  pairs = sorted(pairs, key=lambda pair : pair[0])
  print pairs
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
    if max_growth < sparseness_threshold :
      print pair
      if cluster.intersection(set(pair)) :
        print 'intersects'
        cluster = cluster.union(set(pair))
      elif cluster :
        clusters.append(cluster)
        cluster = set(pair)
      else :
        cluster = set(pair)

  if cluster :
    clusters.append(cluster)

  return clusters
      
if __name__ == '__main__' :

  duplicates = (((1,2), .9),
                ((1,3), .7),
                ((1,4), .2),
                ((2,5), .6),
                ((2,7), .8),
                ((2,3), .9),
                ((3,5), .2)
                )

  nn = nearestNeighbors(duplicates)
  print nn

  neighborhood_attributes = neighborhoodAttributes(nn, 2, 3)

  compact_pairs = compactPairs(neighborhood_attributes)        

  print partition(compact_pairs, neighborhood_attributes, 2)
