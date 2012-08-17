import json
import core
import training_sample
import crossvalidation
from predicates import *
import blocking
import clustering

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

class Dedupe:

  def __init__(self):
    self.data_model = None
    self.training_data = None
    self.training_pairs = None
    self.alpha = 0
    self.predicates = None
    self.blocked_map = None
    self.candidates = None 
    self.dupes = None
  
  def writeSettings(self, file_name) :
    source_predicates = [(predicate[0].__name__,
                          predicate[1])
                         for predicate in self.predicates]
    with open(file_name, 'w') as f :
      json.dump({'data model' : self.data_model,
                 'predicates' : source_predicates}, f)
    
    
  def readSettings(self, file_name) :
    with open(file_name, 'r') as f :
      learned_settings = json.load(f)
  
    self.data_model = learned_settings['data model']
    self.predicates = [(eval(predicate[0]), predicate[1])
                  for predicate in learned_settings['predicates']]
  
  def writeTraining(self, file_name) :
    with open(file_name, 'w') as f :
      json.dump(self.training_pairs, f)
    
    
  def readTraining(self, file_name) :
    with open(file_name, 'r') as f :
      training_pairs_raw = json.load(f)
    
    training_pairs = {0 : [], 1 : []}
    for label, examples in training_pairs_raw.iteritems() :
      for pair in examples :
        training_pairs[int(label)].append((frozendict(pair[0]),
                                           frozendict(pair[1])))
                          
    self.training_pairs = training_pairs
    self.training_data = training_sample.addTrainingData(self.training_pairs, self.data_model)
  
  def activeLearning(self, data_d, labelingFunction, numTrainingPairs = 30) :
    self.training_data, self.training_pairs, self.data_model = training_sample.activeLearning(core.sampleDict(data_d, 700), self.data_model, labelingFunction, numTrainingPairs)
  
  
  def findAlpha(self) :
    self.alpha = crossvalidation.gridSearch(self.training_data,
                                              core.trainModel,
                                              self.data_model,
                                              k = 10)
  
  def train(self, num_iterations = 100) :
    self.data_model = core.trainModel(self.training_data, num_iterations, self.data_model, self.alpha)
  
  def learnBlocking(self, data_d, semi_supervised) :
    if semi_supervised :
      confident_nonduplicates = blocking.semiSupervisedNonDuplicates(core.sampleDict(data_d, 700),
                                                        self.data_model)

      self.training_pairs[0].extend(confident_nonduplicates)
      
    self.predicates = blocking.trainBlocking(self.training_pairs,
                                 (wholeFieldPredicate,
                                  tokenFieldPredicate,
                                  commonIntegerPredicate,
                                  sameThreeCharStartPredicate,
                                  sameFiveCharStartPredicate,
                                  sameSevenCharStartPredicate,
                                  nearIntegersPredicate,
                                  commonFourGram,
                                  commonSixGram),
                                 self.data_model, 1, 1)
                                 
  def mapBlocking(self, data_d, semi_supervised = True) : 
    if (not self.predicates) :
      self.learnBlocking(data_d, semi_supervised)
    
    self.blocked_map = blocking.blockingIndex(data_d, self.predicates)
  
  def identifyCandidates(self) :
    self.candidates = blocking.mergeBlocks(self.blocked_map)
    
  def score(self, data_d) :
    self.dupes = core.scoreDuplicates(self.candidates, data_d, self.data_model)
  
  def duplicateClusters(self, threshold = .5) : 
    return clustering.hierarchical.cluster(self.dupes, threshold)
  
  # def activelearn :
# 
#   def train :
#   
#   def loadSettings :
#   
#   def writeSettings :
#   
#   def blocking_map : #[dictionary where k is a block key, and elements are row_ids]
#   
#   def score :
#   
#   def clusters :
