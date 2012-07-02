import lr
from blocking import trainBlocking, blockingIndex, mergeBlocks, allCandidates
from predicates import *
from math import log, exp
import affinegap
import numpy

# based on field type, calculate using the appropriate distance function and return distance
def calculateDistance(instance_1, instance_2, fields, distances) :

  for i, name in enumerate(fields) :
    if fields[name]['type'] == 'String' :
      distanceFunc = affinegap.normalizedAffineGapDistance

    distances[0]['names'][i] = name
    distances[0]['values'][i] = distanceFunc(instance_1[name],instance_2[name], -5, 5, 4, 1, .125)

  return distances

# using logistic regression, train weights for all fields in the data model
def trainModel(training_data, iterations, data_model) :
    trainer = lr.LogisticRegression()
    trainer.train(training_data, iterations)

    data_model['bias'] = trainer.bias
    for name in data_model['fields'] :
        data_model['fields'][name]['weight'] = trainer.weight[name]

    return(data_model)

# assign a score of how likely a pair of records are duplicates
def recordDistances(candidates, data_d, data_model) :
  # The record array has two elements, the first element is an array
  # of floats that has length equal the number of fields. The second
  # argument is a array of length 2 which stores the id of the
  # considered elements in the pair.

  fields = data_model['fields']

  field_dtype = [('names', 'a10', (len(fields)),),
                 ('values', 'f4', (len(fields)),)
                 ]
  
  distances = numpy.zeros(1, dtype=field_dtype)

  record_dtype = [('pairs', [('pair1', 'i4'),
                             ('pair2', 'i4')]),
                  ('field_distances', field_dtype)
                  ]

  record_distances = numpy.zeros(len(candidates), dtype=record_dtype)


#  print record_distance

  for i, pair in enumerate(candidates) :
    
    c_distances = calculateDistance(data_d[pair[0]],
                                    data_d[pair[1]],
                                    fields,
                                    distances)

    record_distances[i] = ((pair[0], pair[1]),
                           (c_distances['names'],
                            c_distances['values'])
                           )
    
  return record_distances  

def scorePairs(record_distances, data_model) :
  fields = data_model['fields']

  field_weights = [fields[name]['weight'] for name in fields]
  bias = data_model['bias']



  #field_distances = [col for row in record_distances['field_distances']
  #                   for col in row]
  field_distances = record_distances['field_distances']['values']

#  field_distances = numpy.reshape(field_distances, (-1, len(fields)))

  scores = numpy.dot(field_distances, field_weights)

  scores = numpy.exp(scores + bias)/(1 + numpy.exp(scores + bias))

  return(scores)

# identify all pairs above a set threshold as duplicates
def findDuplicates(candidates, data_d, data_model, threshold) :

  record_distances = recordDistances(candidates, data_d, data_model)
  duplicateScores = scorePairs(record_distances, data_model)

  return([pair for pair in duplicateScores if pair.values()[0] > threshold])
  
# define a data type for hashable dictionaries
class frozendict(dict):
    def _blocked_attribute(obj):
        raise AttributeError, "A frozendict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args):
        new = dict.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args):
        pass

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(tuple(sorted(self.items())))
            return h

    def __repr__(self):
        return "frozendict(%s)" % dict.__repr__(self)

# main execution of dedupe
if __name__ == '__main__':
  from canonical_example import init

  # user defined function to label pairs as duplicates or non-duplicates
  def consoleLabel(uncertain_pairs) :
    duplicates = []
    nonduplicates = []
    
    for pair in uncertain_pairs :
      label = ''
      
      for instance in tuple(pair) :
        print instance.values()
        print "Do these records refer to the same thing?"  
      
        valid_response = False
        while not valid_response :
          label = raw_input('yes(y)/no(n)/unsure(u)\n')
          if label in ['y', 'n', 'u'] :
            valid_response = True

        if label == 'y' :
          duplicates.append(pair)
        elif label == 'n' :
          nonduplicates.append(pair)
        elif label != 'u' :
          print 'Nonvalid response'
          raise

    return({0:nonduplicates, 1:duplicates})

  
  num_training_dupes = 200
  num_training_distinct = 16000
  numIterations = 100

  import time
  t0 = time.time()
  (data_d, duplicates_s, data_model) = init()
  #candidates = allCandidates(data_d)
  #print "training data: "
  #print duplicates_s

  print "number of duplicates pairs"
  print len(duplicates_s)
  print ""

  #lets do some active learning here
  training_pairs = activeLearning(data_d, data_model, consoleLabel);
  
  
  
  #training_pairs = randomTrainingPairs(data_d,
  #                                     duplicates_s,
  #                                     num_training_dupes,
  #                                     num_training_distinct)

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



  print ""
  print "Blocking reduced the number of comparisons by",
  print int((1-len(candidates)/float(0.5*len(data_d)**2))*100),
  print "%"
  print "We'll make",
  print len(candidates),
  print "comparisons."

  training_data = trainingDistances(training_pairs, data_model)
  #print "training data from known duplicates: "
  #for instance in training_data :
  #  print instance

  print ""
  print "number of training items: "
  print len(training_data)
  print ""

  print "training weights ..."
  data_model = trainModel(training_data, numIterations, data_model)
  print ""

  print "Learned Weights"
  for k1, v1 in data_model.items() :
    try:
      for k2, v2 in v1.items() :
        print (k2, v2['weight'])
    except :
      print (k1, v1)

  print ""
  
  print "finding duplicates ..."
  print ""
  dupes = findDuplicates(candidates, data_d, data_model, .60)

  dupe_ids = set([frozenset(list(dupe_pair.keys()[0])) for dupe_pair in dupes])
  true_positives = dupe_ids & duplicates_s
  false_positives = dupe_ids - duplicates_s
  uncovered_dupes = duplicates_s - dupe_ids

  print "False negatives" 
  for pair in uncovered_dupes :
         print ""
         for instance in tuple(pair) :
           print data_d[instance].values()

  print "____________________________________________"
  print "False positives" 

  for pair in false_positives :
    print ""
    for instance in tuple(pair) :
      print data_d[instance].values()

  print ""

  print "found duplicate"
  print len(dupes)

  print "precision"
  print (len(dupes) - len(false_positives))/float(len(dupes))

  print "recall"
  print  len(true_positives)/float(len(duplicates_s))
  print "ran in ", time.time() - t0, "seconds"

  print findUncertainPairs(data_d, data_model, 10)
