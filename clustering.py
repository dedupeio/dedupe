duplicates = {(1,2) : .9,
              (1,3) : .7,
              (1,4) : .2,
              (2,5) : .6,
              (2,3) : .9,
              (3,5) : .2
              }

print duplicates

def nearestNeighbors(duplicates) :
  nn = {}
  for pair in duplicates :
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

nn = nearestNeighbors(duplicates)
print nn

print neighborhoodAttributes(nn, 2, 4)
    
        
