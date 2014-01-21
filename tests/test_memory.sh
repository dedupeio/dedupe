valgrind --tool=massif --suppressions=/usr/share/doc/python26-devel-2.6.8/valgrind-python.supp --massif-out-file=out.txt --depth=1 python2.6 tests/test_affine_memory.py
ms_print out.txt  | less
