import random
import shelve
import pickle
import os
import tempfile

fd, file_path = tempfile.mkstemp()
os.close(fd)

try:
    shelf = shelve.open(file_path, 'n',
                        protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    if 'db type could not be determined' in str(e):
        os.remove(file_path)
        shelf = shelve.open(file_path, 'n',
                            protocol=pickle.HIGHEST_PROTOCOL)


record = {'foo' : 'bar'}
ids = set(range(1000))
        
for i in range(10000):
    k = str(random.randrange(1000))
    if k in shelf:
        shelf[k] += [(i, record, ids)]
    else:
        shelf[k] = [(i, record, ids)]

for block in shelf.values():
    if len(block) > 1:
        print(block)

