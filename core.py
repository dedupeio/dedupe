import lr
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
def scoreDuplicates(candidates, data_d, data_model, threshold = None) :

  record_distances = recordDistances(candidates, data_d, data_model)
  duplicate_scores = scorePairs(record_distances, data_model)

  scored_pairs = zip(candidates, duplicate_scores)
  if (threshold) :
    return([pair for pair in scored_pairs if pair[1] > threshold])
  else :
    return scored_pairs
  
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

