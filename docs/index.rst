.. dedupe documentation master file, created by
   sphinx-quickstart on Thu Apr 10 11:27:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dedupe's documentation!
==================================

Contents:

.. toctree::
   :maxdepth: 2

   API-documentation  
   Choosing-a-good-threshold   



*Dedupe is an open source python library that quickly de-duplicates
large sets of data.*

Deduplication, entity resolution, record linkage, author
disambiguation, and others ...

As different research communities encountered this problem, they each
gave it a new name but, ultimately, its all about trying to figure out
what records are referring to the same thing.

   Bibliography.rst
   Matching-records.rst
   Choosing-a-good-threshold.rst  
   OSX-Install-Notes.rst
   Grouping-duplicates.rst        
   Special-Cases.rst
   Home.rst             
   Making-smart-comparisons.rst


Features
========
 * **machine learning** - reads in human labeled data to automatically create optimum weights and blocking rules
 * **runs on a laptop** - makes intelligent comparisons so you don't need a powerful server to run it
 * **built as a library** - so it can be integrated in to your applications or import scripts
 * **extensible** - supports adding custom data types, string comparators and blocking rules
 * **open source** - anyone can use, modify or add to it

Installation
============
pip install numpy
pip install dedupe


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

