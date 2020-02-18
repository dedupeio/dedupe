import numpy
import warnings
import logging

logger = logging.getLogger(__name__)

MULTIPROCESSING = True
# Deal with Mac OS X issuse
config_info = str([value for key, value in
                   numpy.__config__.__dict__.items()
                   if key.endswith("_info")]).lower()

if "accelerate" in config_info or "veclib" in config_info:
    warnings.warn("NumPy linked against 'Accelerate.framework'. "
                  "Multiprocessing will be disabled."
                  " http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html")
    MULTIPROCESSING = False
elif "mkl_core" in config_info:
    warnings.warn("Numpy linked against MKL. "
                  "Multiprocessing will be disabled. https://github.com/joblib/joblib/issues/138")
    MULTIPROCESSING = False

if MULTIPROCESSING:
    from multiprocessing import Process, Pool, Queue
    from multiprocessing import SimpleQueue
else:
    from multiprocessing.dummy import Process, Pool, Queue  # type: ignore # noqa: F401
    SimpleQueue = Queue  # type: ignore
