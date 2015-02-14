#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError :
    raise ImportError("setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools")

install_requires=['numpy>=1.9', 
                  'fastcluster', 
                  'hcluster',
                  'categorical-distance',
                  'rlr',
                  'affinegap',
                  'canonicalize',
                  'simplecosine',
                  'haversine',
                  'BTrees==4.0.8',
                  'zope.interface', 
                  'zope.index']

try:
    from json.scanner import py_make_scanner
except ImportError:
    install_requires.append('simplejson')

try : 
    from collections import OrderedDict
except ImportError:
    install_requires.append('ordereddict')


setup(
    name='dedupe',
    url='https://github.com/datamade/dedupe',
    version='0.7.7.0.4',
    description='A python library for accurate and scaleable data deduplication and entity-resolution',
    packages=['dedupe', 'dedupe.variables'],
    ext_modules=[Extension('dedupe.cpredicates', ['src/cpredicates.c'])],
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
    *dedupe is a library that uses machine learning to perform de-duplication and entity resolution quickly on structured data.*

    **dedupe** will help you: 

    * **remove duplicate entries** from a spreadsheet of names and addresses
    * **link a list** with customer information to another with order history, even without unique customer id's
    * take a database of campaign contributions and **figure out which ones were made by the same person**, even if the names were entered slightly differently for each record

    dedupe takes in human training data and comes up with the best rules for your dataset to quickly and automatically find similar records, even with very large databases.
    
    Important links:
    
    * Documentation: http://dedupe.rtfd.org/
    * Repository: https://github.com/datamade/dedupe
    * Issues: https://github.com/datamade/dedupe/issues
    * Examples: https://github.com/datamade/dedupe-examples
    """
    )
