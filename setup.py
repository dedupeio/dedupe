#!/usr/bin/python
# -*- coding: utf-8 -*-
from distutils.core import setup, Extension
import numpy

setup(
    name='Dedupe',
    url='https://github.com/open-city/dedupe',
    version='0.3',
    packages=['dedupe', 'dedupe.clustering'],
    ext_modules=[Extension('dedupe.affinegap', ['src/affinegap.c']),
                 Extension('dedupe.lr',
                           sources=['src/lr.c'],
                           include_dirs=[numpy.get_include()])],
    license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
    install_requires=['numpy', 'fastcluster', 'hcluster', 'networkx'],
    long_description=open('README.md').read(),
    )
