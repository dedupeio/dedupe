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

install_requires=['numpy', 
                  'fastcluster', 
                  'hcluster', 
                  'networkx', 
                  'zope.interface', 
                  'zope.index']

try:
    import json
except ImportError:
    install_requires.append('simplejson')


setup(
    name='dedupe',
    url='https://github.com/datamade/dedupe',
    version='0.5.0.5.2',
    description='A python library for accurate and scaleable data deduplication and entity-resolution',
    packages=['dedupe', 'dedupe.distance'],
    ext_modules=[NumpyExtension('dedupe.distance.affinegap', ['src/affinegap.c']),
                 Extension('dedupe.distance.jaccard', ['src/jaccard.c']),
                 NumpyExtension('dedupe.distance.haversine', ['src/haversine.c'], libraries=['m']),
                 NumpyExtension('dedupe.distance.cosine', ['src/cosine.c'], libraries=['m']),
                 NumpyExtension('dedupe.lr', sources=['src/lr.c'])],

                 

    license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Cython', 
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    long_description="""
    *Dedupe is an open source python library that quickly de-duplicates
    large sets of data.*
    
    Deduplication, entity resolution, record linkage, author
    disambiguation, and others ...
    
    As different research communities encountered this problem, they each
    gave it a new name but, ultimately, its all about trying to figure out
    what records are referring to the same thing.
    
    Important links:
    
    * Documentation: http://dedupe.rtfd.org/
    * Repository: https://github.com/datamade/dedupe
    * Issues: https://github.com/datamade/dedupe/issues
    """
    )
