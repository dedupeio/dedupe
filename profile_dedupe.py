import profile
import dedupe

#import timeit
#t = timeit.Timer(setup='from __main__ import doit1', stmt='doit1()')
#t.timeit()

def testProfile() :
  data_d, header, duplicates_s = dedupe.canonicalImport("./datasets/restaurant-nophone-training.csv")
  data_model = dedupe.dataModel()
  candidates = dedupe.identifyCandidates(data_d)

  training_data = dedupe.createTrainingData(data_d, duplicates_s, 10, data_model)

profile.run("testProfile()")