#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError :
    raise ImportError("setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools")


# from Michael Hoffman's http://www.ebi.ac.uk/~hoffman/software/sunflower/

class NumpyExtension(Extension):

    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)

        self._include_dirs = self.include_dirs
        del self.include_dirs  # restore overwritten property

    # warning: Extension is a classic class so it's not really read-only

    @property
    def include_dirs(self):
        from numpy import get_include

        return self._include_dirs + [get_include()]

setup(
    name='Dedupe',
    url='https://github.com/open-city/dedupe',
    version='0.4',
    packages=['dedupe', 'dedupe.distance'],
    ext_modules=[NumpyExtension('dedupe.distance.affinegap', ['src/affinegap.c']),
                 Extension('dedupe.distance.jaccard', ['src/jaccard.c']),
                 NumpyExtension('dedupe.distance.haversine', ['src/haversine.c'], libraries=['m']),
                 NumpyExtension('dedupe.distance.cosine', ['src/cosine.c'], libraries=['m']),
                 NumpyExtension('dedupe.lr', sources=['src/lr.c'])],
    license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
    install_requires=['numpy', 'fastcluster', 'hcluster', 'networkx'],
    long_description=open('README.md').read(),
    )
