#!/usr/bin/python
# -*- coding: utf-8 -*-
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from dedupe._init import *  # noqa


import os
module_path = os.path.dirname(os.path.abspath(__file__))
print(f'*** Custom Conjura dedupe fork loaded: {module_path} ***')
