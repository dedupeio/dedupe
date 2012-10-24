import collections
import math
import re

def createTfidfPredicate(idf_dictionary, threshold) :
  def tfidfPredicate(string) :
    tokens = get_tokens(string)
    tf = collections.Counter(tokens)
    filtered_tokens = [token for token in tokens 
                       if tf[token] * idf_dictionary[token] > threshold] 

    return filtered_tokens

  return tfidfPredicate

def documentFrequency(corpus) : 
  num_docs = 0
  term_num_docs = collections.defaultdict(int)
  num_docs = len(corpus)

  for doc in corpus :
    tokens = get_tokens(doc)
    for token in set(tokens) :
      term_num_docs[token] += 1

  for term, count in term_num_docs.iteritems() :
    term_num_docs[term] = math.log((num_docs + 0.5) / (float(count) + 0.5))

  term_num_docs_default = collections.defaultdict(lambda: math.log((num_docs + 0.5)/0.5))     # term : num_docs_containing_term
  term_num_docs_default.update(term_num_docs)

  return term_num_docs_default

def get_tokens(str):
  """Break a string into tokens, preserving URL tags as an entire token.

     This implementation does not preserve case.  
     Clients may wish to override this behavior with their own tokenization.
  """
  return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())

# corpus = ["Forest is cool and stuff", "Derek is cool and maybe other stuff"]
# a = documentFrequency(corpus)
# print a
# print a['foo']