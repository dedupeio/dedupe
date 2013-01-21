import numpy
import gc

#@profile
def testing():
  a = numpy.array([1]*1000000, 'f4')
  return a

def testing_list():
  a = [1]*1000000
  return a

def call_testing():
  print testing()
  gc.collect()
  #testing_list()
  #gc.collect()



call_testing()