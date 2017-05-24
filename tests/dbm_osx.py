import pickle
import shelve
import tempfile
import os

with open('tests/blocks_d.pickle', 'rb') as f:
    blocks_d = pickle.load(f)


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


for k, v in blocks_d.items():
    shelf[k] = v

for block in shelf.values():
    if len(block) > 1:
        print(block[0])
