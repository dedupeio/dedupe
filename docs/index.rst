.. dedupe documentation master file, created by
   sphinx-quickstart on Thu Apr 10 11:27:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================
Dedupe |release|
================

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

Contents:

.. toctree::
   :maxdepth: 1

   API-documentation
   Field-definition
   OSX-Install-Notes
   Home             


Features
========
 * **machine learning** - reads in human labeled data to automatically create optimum weights and blocking rules
 * **runs on a laptop** - makes intelligent comparisons so you don't need a powerful server to run it
 * **built as a library** - so it can be integrated in to your applications or import scripts
 * **extensible** - supports adding custom data types, string comparators and blocking rules
 * **open source** - anyone can use, modify or add to it

Installation
============

Dedupe requires `numpy <http://numpy.scipy.org/>`__, which can be
complicated to install.  If you are installing numpy for the first
time, `follow these instructions
<http://docs.scipy.org/doc/numpy/user/install.html>`__. You'll need to
version 1.6 of numpy or higher.


.. code-block:: bash

   pip install "numpy>=1.6"
   pip install dedupe

Mac OS X Install Notes
----------------------

Before installing, you may need to set the following environmental
variables from the command line

.. code-block:: bash

   export CFLAGS=-Qunused-arguments 
   export CPPFLAGS=-Qunused-arguments

With default configurations, dedupe cannot do parallel processing on Mac OS X.
:doc:`Read about instructions on how to enable this <OSX-Install-Notes>`.

Using dedupe
============

Dedupe is a library and not a stand-alone command line tool. To
demonstrate its usage, we have come up with a `few example recipes for
different sized datasets for you
<https://github.com/datamade/dedupe-examples/archive/0.5.zip>`__
(`repo <https://github.com/datamade/dedupe-examples>`__, as well as
annotated source code:

* `Small data deduplication <http://datamade.github.com/dedupe/doc/csv_example.html>`__
* `Bigger data deduplication ~700K <http://datamade.github.com/dedupe/doc/mysql_example.html>`__
* `Record Linkage  <http://datamade.github.com/dedupe/doc/record_linkage_example.html>`__

Errors / Bugs
=============

If something is not behaving intuitively, it is a bug, and should be
reported. `Report it here <https://github.com/datamade/dedupe/issues>`__

Contributing to dedupe
======================

Check out `dedupe <https://github.com/datamade/dedupe>`__
repo for how to contribute to the library.

Check out `dedupe-examples
<https://github.com/datamade/dedupe-examples>`__ for how to contribute
a useful example of using dedupe.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

