#!/usr/bin/python
# -*- coding: utf-8 -*-
from distutils.core import setup, Extension

setup(
    name='Dedupe',
    version='0.3',
    packages=['dedupe', 'dedupe.clustering'],
    ext_modules=[Extension('dedupe.affinegap', ['src/affinegap.c'])],
    install_requires=['numpy', 'fastcluster', 'hcluster'],
    )
