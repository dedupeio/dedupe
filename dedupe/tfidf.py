import collections
import math
import re
import core

class TfidfPredicate(float):
  def __new__(self, threshold) :
    return float.__new__(self, threshold)
  
  def __init__(self, threshold) :
    self.__name__ = "TF-IDF:" + str(threshold)



def coverage(threshold, field, training_pairs, inverted_index) :

  docs = set(instance for pair in training_pairs for instance in pair)

  corpus = dict((i, doc[field]) for i, doc in enumerate(docs))
  id_lookup = dict((instance, i) for i, instance in enumerate(docs))

  blocked_data = createCanopies(corpus, inverted_index, threshold)

  coverage_dict = {}

  for pair in training_pairs:
    id_pair = set(id_lookup[instance] for instance in pair)
    
    if any(id_pair.issubset(block) for block in blocked_data) :
      coverage_dict[pair] = 1
    else:
      coverage_dict[pair] = -1

  return coverage_dict


def documentFrequency(corpus) : 
  num_docs = 0
  term_num_docs = collections.defaultdict(int)
  num_docs = len(corpus)
  stop_word_threshold = num_docs * 1
  stop_words = []

  for doc_id, doc in corpus.iteritems() :
    tokens = getTokens(doc)
    for token in set(tokens) :
      term_num_docs[token] += 1

  for term, count in term_num_docs.iteritems() :
    if count < stop_word_threshold :
      term_num_docs[term] = math.log((num_docs + 0.5) / (float(count) + 0.5))
    else:
      term_num_docs[term] = 0
      stop_words.append(term)

  if stop_words:
    print 'stop words:', stop_words

  term_num_docs_default = collections.defaultdict(lambda: math.log((num_docs + 0.5)/0.5))    # term : num_docs_containing_term
  term_num_docs_default.update(term_num_docs)

  return term_num_docs_default


def invertedIndex(corpus):
  inverted_index = collections.defaultdict(list)
  for doc_id, doc in corpus.iteritems() :
    tokens = getTokens(doc)
    for token in set(tokens) :
      inverted_index[token].append(doc_id)

  return inverted_index

def getTokens(str):
  return str.lower().split()

def createCanopies(corpus_original, df_index, threshold) :
  blocked_data = []
  seen_set = set([])
  corpus = corpus_original.copy()
  inverted_index = invertedIndex(corpus)
  while corpus :
    doc_id, center = corpus.popitem()
    if not center :
      continue
    
    seen_set.add(doc_id)
    block = [doc_id]
    candidate_set = set([])
    tokens = getTokens(center)
    center_dict = tfidfDict(center, df_index)

    for token in tokens :
      candidate_set.update(inverted_index[token])

    candidate_set = candidate_set - seen_set
    for doc_id in candidate_set :
      candidate_dict = tfidfDict(corpus[doc_id], df_index)
      similarity = cosineSimilarity(candidate_dict, center_dict)

      if similarity > threshold :
        block.append(doc_id)
        seen_set.add(doc_id)
        del corpus[doc_id]

    if len(block) > 1 :
      blocked_data.append(block)

  return blocked_data

def cosineSimilarity(doc_dict_1, doc_dict_2) :
  common_keys = set(doc_dict_1.keys()) & set(doc_dict_2.keys())
  dot_product = sum(doc_dict_1[key] * doc_dict_2[key] for key in common_keys)

  if dot_product == 0 :
    return 0

  else:
    norm_1 = calculateNorm(doc_dict_1)
    norm_2 = calculateNorm(doc_dict_2)

    return dot_product / (norm_1 * norm_2)

def calculateNorm(doc_dict) :
  norm = sum(value*value for value in doc_dict.values())
  return math.sqrt(norm) 

def tfidfDict(doc, df_index) :
  tokens = getTokens(doc)
  return dict((token, tokens.count(token) * df_index[token]) for token in set(tokens))

