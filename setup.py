#!/usr/bin/python
# -*- coding: utf-8 -*-
from distutils.core import setup, Extension
import numpy

setup(
    name='Dedupe',
    version='0.3',
    packages=['dedupe', 'dedupe.clustering'],
    ext_modules=[Extension('dedupe.affinegap', ['src/affinegap.c']),
                 Extension('dedupe.lr',
                           sources=['src/lr.c'],
                           include_dirs=[numpy.get_include()])],
    install_requires=['numpy', 'fastcluster', 'hcluster'],
    )
