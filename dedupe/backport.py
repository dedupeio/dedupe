import numpy
import weakref
import threading
import warnings
import platform
import sys
import inspect
from collections import OrderedDict

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


# This backport code comes from https://gerrit.wikimedia.org/r/#/c/229683/1/pywikibot/tools/__init__.py,cm
# (C) Pywikibot team, 2008-2015
#
# Distributed under the terms of the MIT license.

try:
    Parameter = inspect.Parameter
    Signature = inspect.Signature

    def signature(obj):
        """Return the signature for callable and None if a ValueError occurs."""
        try:
            return inspect.signature(obj)
        except ValueError:
            return None
except:
    # A basic implementation of the Parameter and Signature class to provide
    # the same support for getargspec on versions before 3.3

    class _empty:

        """A dummy class to show that the Parameter has no default value."""

        pass

    class Parameter(object):

        """A basic Parameter class implementation."""

        POSITIONAL_ONLY = 0
        POSITIONAL_OR_KEYWORD = 1
        VAR_POSITIONAL = 2
        KEYWORD_ONLY = 3
        VAR_KEYWORD = 4

        empty = _empty

        def __init__(self, name, kind, default=_empty):
            """Constructor."""
            self.name = name
            self.kind = kind
            self.default = default

        def replace(self, **kwargs):
            """Create a copy and replace the given variables."""
            name = kwargs.pop('name', self.name)
            kind = kwargs.pop('kind', self.kind)
            default = kwargs.pop('default', self.default)
            assert not kwargs
            return Parameter(name, kind, default=default)

    class Signature(object):

        """A basic Signature class implementation."""

        def __init__(self, parameters=None):
            """Constructor."""
            self.parameters = OrderedDict()
            for param in parameters or []:
                self.parameters[param.name] = param

    def signature(obj):
        """Return the basic signature for the callable."""
        try:
            spec = inspect.getargspec(obj)
        except TypeError:
            return None
        defaults = (_empty,) * len(spec[0])
        if spec[3] is not None:
            # spec[3] are the defaults counting from the end, add to the
            # "non-existing" defaults and shift by number of params
            defaults = (defaults + spec[3])[-len(spec[0]):]
        parameters = []
        for arg, default in zip(spec[0], defaults):
            parameters += [Parameter(arg, Parameter.POSITIONAL_OR_KEYWORD,
                                     default=default)]
        if spec[1] is not None:
            parameters += [Parameter(spec[1], Parameter.VAR_POSITIONAL)]
        if spec[2] is not None:
            parameters += [Parameter(spec[2], Parameter.VAR_KEYWORD)]
        return Signature(parameters)

