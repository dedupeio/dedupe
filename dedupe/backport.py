import numpy
import weakref
import threading
import warnings
import platform
import sys

from future.utils import viewitems
from future.builtins import range

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

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
    1-D arrays to form the cartesian product of.
    out : ndarray
    Array to place the cartesian product in.
    
    Returns
    -------
    out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.
    
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
    [1, 4, 7],
    [1, 5, 6],
    [1, 5, 7],
    [2, 4, 6],
    [2, 4, 7],
    [2, 5, 6],
    [2, 5, 7],
    [3, 4, 6],
    [3, 4, 7],
    [3, 5, 6],
    [3, 5, 7]])
    
    References
    ----------
    http://stackoverflow.com/q/1208118
    
    """
    arrays = [numpy.asarray(x).ravel() for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.empty([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

