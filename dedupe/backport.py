import numpy
import weakref
import threading
import warnings
import platform
import sys

from future.utils import viewitems

MULTIPROCESSING = True
# Deal with Mac OS X issuse
config_info = str([value for key, value in
                   viewitems(numpy.__config__.__dict__)
                   if key.endswith("_info")]).lower()

if "accelerate" in config_info or "veclib" in config_info :
    warnings.warn("NumPy linked against 'Accelerate.framework'. "
                  "Multiprocessing will be disabled."
                  " http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html")
    MULTIPROCESSING = False
elif platform.system() == 'Windows' :
    warnings.warn("Dedupe does not currently support multiprocessing on Windows")
    MULTIPROCESSING = False

if MULTIPROCESSING :        
    from multiprocessing import Process, Pool, Queue
    if sys.version < '3':
        from multiprocessing.queues import SimpleQueue
    else :
        from multiprocessing import SimpleQueue
else :
    if not hasattr(threading.current_thread(), "_children"): 
        threading.current_thread()._children = weakref.WeakKeyDictionary()
    from multiprocessing.dummy import Process, Pool, Queue
    SimpleQueue = Queue


