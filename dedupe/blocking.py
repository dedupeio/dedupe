from collections import defaultdict
from itertools import product, chain, combinations
from math import sqrt, log
import core
from random import sample

def predicateCoverage(pairs, predicates) :
  coverage = defaultdict(list)
  for pair in pairs :
    for predicate in predicates :
      keys1 = set(product(*[F(pair[0][field])
                       for F, field in predicate]))
      keys2 = set(product(*[F(pair[1][field])
                           for F, field in predicate]))
      if keys1 & keys2 :
        coverage[predicate].append(pair)



  return(coverage)


# Approximate learning of blocking following the ApproxRBSetCover from
# page 102 of Bilenko
def trainBlocking(training_pairs, predicates, data_model, eta, epsilon) :

  sample_size = 1000
  if len(training_pairs) <= sample_size :
    training_distinct = training_pairs[0][:]
  else :
    training_distinct = sample(training_pairs[0][:], sample_size)
    
  training_dupes = training_pairs[1][:]
  n_training_dupes = len(training_dupes)
  n_training_distinct = len(training_distinct)
  print n_training_dupes
  print n_training_distinct
  sample_size = n_training_dupes + n_training_distinct

  fields = [field for field in data_model['fields'] 
            if data_model['fields'][field]['type'] != 'Interaction']
  
  # The set of all predicate functions operating over all fields
  predicateSet = list(product(predicates, fields))

  disjunctive_predicates = list(combinations(predicateSet, 2))
  # filter out disjunctive predicates that operate on same field
  disjunctive_predicates = [predicate for predicate
                            in disjunctive_predicates
                            if predicate[0][1] != predicate[1][1]]

  predicateSet = [(predicate,) for predicate in predicateSet]
  predicateSet.extend(disjunctive_predicates)
  n_predicates = len(predicateSet)

  
  found_dupes = predicateCoverage(training_dupes,
                                  predicateSet)
  found_distinct = predicateCoverage(training_distinct,
                                     predicateSet)


  predicateSet = found_dupes.keys() 

  # We want to throw away the predicates that puts together too many
  # distinct pairs
  eta = sample_size * eta

  [predicateSet.remove(predicate)
   for predicate in found_distinct
   if len(found_distinct[predicate]) >= eta]

  # We don't want to penalize a blocker if it puts distinct pairs
  # together that look like they could be duplicates. Here we compute
  # the expected number of predicates that will cover a duplicate pair
  # We'll remove all the distince pairs from consideration if they are
  # covered by many predicates
  expected_dupe_cover = sqrt(n_predicates / log(n_training_dupes))

  predicate_count = defaultdict(int)
  for pair in chain(*found_distinct.values()) :
      predicate_count[pair] += 1

  training_distinct = [pair for pair in training_distinct
                       if predicate_count[pair] < expected_dupe_cover]


  found_distinct = predicateCoverage(training_distinct,
                                     predicateSet)

  # Greedily find the predicates that, at each step, covers the most
  # duplicates and covers the least distinct pairs, dute to Chvatal, 1979
  finalPredicateSet = []
  print "Uncovered dupes"
  print n_training_dupes
  while n_training_dupes >= epsilon :
        
    optimumCover = 0
    bestPredicate = None
    for predicate in predicateSet :
      try:  
          cover = (len(found_dupes[predicate])
                   / float(len(found_distinct[predicate]))
                   )
      except ZeroDivisionError:
          cover = len(found_dupes[predicate])

      if cover > optimumCover :
        optimumCover = cover
        bestPredicate = predicate


    if not bestPredicate :
        print "Ran out of predicates"
        break

    predicateSet.remove(bestPredicate)
    n_training_dupes -= len(found_dupes[bestPredicate])
    [training_dupes.remove(pair) for pair in found_dupes[bestPredicate]]
    found_dupes = predicateCoverage(training_dupes,
                                    predicateSet)

    print n_training_dupes

    finalPredicateSet.append(bestPredicate)
    
  print "FINAL PREDICATE SET!!!!"
  print finalPredicateSet
  
  if finalPredicateSet :
    return finalPredicateSet
  else :
    print "No predicate found!"
    raise 


def blockingIndex(data_d, predicates) :
  blocked_data = defaultdict(set)
  for key, instance in data_d.items() :
    for predicate in predicates :
      predicate_tuples = product(*[F(data_d[key][field])
                                  for F, field in predicate])
      
      for predicate_tuple in predicate_tuples :
        blocked_data[str(predicate_tuple)].add(key)

 
  return blocked_data

def mergeBlocks(blocked_data) :
  candidates = set()
  for block in blocked_data.values() :
    if len(block) > 1 :
      block = sorted(block)
      for pair in combinations(block, 2) :
        candidates.add(pair)
    
  return candidates

def allCandidates(data_d) :
  return list(combinations(sorted(data_d.keys()),2))

def semiSupervisedNonDuplicates(data_d,
                                data_model,
                                nonduplicate_confidence_threshold = .7) :
  
  #this is an expensive call and we're making it multiple times
  pairs = allCandidates(data_d)
  record_distances = core.recordDistances(pairs, data_d, data_model)
  
  confident_nonduplicate_ids = []
  scored_pairs = core.scorePairs(record_distances, data_model)
  for i, score in enumerate(scored_pairs) :
    if score < (1 - nonduplicate_confidence_threshold) : 
      confident_nonduplicate_ids.append(record_distances['pairs'][i]) 
  
  confident_nonduplicate_pairs = [(data_d[pair[0]], data_d[pair[1]])
                                  for pair in confident_nonduplicate_ids]

  return confident_nonduplicate_pairs


if __name__ == '__main__':
  from dedupe import randomTrainingPairs
  from test_data import init
  from predicates import *


  numTrainingPairs = 64000
  (data_d, duplicates_s, data_model) = init()

  training_pairs = randomTrainingPairs(data_d,
                                       duplicates_s,
                                       numTrainingPairs)

  predicates = trainBlocking(training_pairs,
                             (wholeFieldPredicate,
                              tokenFieldPredicate,
                              commonIntegerPredicate,
                              sameThreeCharStartPredicate,
                              sameFiveCharStartPredicate,
                              sameSevenCharStartPredicate,
                              nearIntegersPredicate,
                              commonFourGram,
                              commonSixGram),
                             data_model, 1, 1)  

  blocked_data = blockingIndex(data_d, predicates)
  candidates = mergeBlocks(blocked_data)
  print len(candidates)
