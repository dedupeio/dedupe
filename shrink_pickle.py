import pickle
import objgraph

with open('active_learner.pickle', 'rb') as f:
    active_learner = pickle.load(f)

active_learner.pairs[:] = []
active_learner.candidates[:] = []
#active_learner.blocker = None
active_learner.classifier = None
active_learner.learners = None

with open('active_learner_1.pickle', 'wb') as f:
    pickle.dump(active_learner, f)

#objgraph.show_refs([active_learner], max_depth=5)
