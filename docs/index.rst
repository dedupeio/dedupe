.. dedupe documentation master file, created by
   sphinx-quickstart on Thu Apr 10 11:27:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================
Dedupe |release|
================

*dedupe is a library that uses machine learning to perform de-duplication and entity resolution quickly on structured data.*

If you're looking for the documentation for the Dedupe.io Web API, you can find that here: https://apidocs.dedupe.io/

**dedupe** will help you: 

* **remove duplicate entries** from a spreadsheet of names and addresses
* **link a list** with customer information to another with order history, even without unique customer id's
* take a database of campaign contributions and **figure out which ones were made by the same person**, even if the names were entered slightly differently for each record

dedupe takes in human training data and comes up with the best rules for your dataset to quickly and automatically find similar records, even with very large databases.

Important links
===============

* Documentation: https://docs.dedupe.io/
* Repository: https://github.com/dedupeio/dedupe
* Issues: https://github.com/dedupeio/dedupe/issues
* Mailing list: https://groups.google.com/forum/#!forum/open-source-deduplication
* Examples: https://github.com/dedupeio/dedupe-examples
* IRC channel, `#dedupe on irc.freenode.net <http://webchat.freenode.net/?channels=dedupe>`__

Tools built with dedupe
=======================

`Dedupe.io <https://dedupe.io/>`__
A full service web service powered by dedupe for de-duplicating and find matches in your messy data. It provides an easy-to-use interface and provides cluster review and automation, as well as advanced record linkage, continuous matching and API integrations. `See the product page <https://dedupe.io/>`__ and the `launch blog post <https://datamade.us/blog/introducing-dedupeio>`__.

`csvdedupe <https://github.com/dedupeio/csvdedupe>`__
Command line tool for de-duplicating and `linking <https://github.com/dedupeio/csvdedupe#csvlink-usage>`__ CSV files. Read about it on `Source Knight-Mozilla OpenNews <https://source.opennews.org/en-US/articles/introducing-cvsdedupe/>`__.

Contents
========

.. toctree::
   :maxdepth: 1

   API-documentation
   Variable-definition
   How-it-works
   Bibliography


Features
========
 * **machine learning** - reads in human labeled data to automatically create optimum weights and blocking rules
 * **runs on a laptop** - makes intelligent comparisons so you don't need a powerful server to run it
 * **built as a library** - so it can be integrated in to your applications or import scripts
 * **extensible** - supports adding custom data types, string comparators and blocking rules
 * **open source** - anyone can use, modify or add to it

Installation
============

.. code-block:: bash

   pip install dedupe

Using dedupe
============

Dedupe is a library and not a stand-alone command line tool. To
demonstrate its usage, we have come up with a `few example recipes for
different sized datasets for you
<https://github.com/dedupeio/dedupe-examples/archive/0.5.zip>`__
(`repo <https://github.com/dedupeio/dedupe-examples>`__, as well as
annotated source code:

* `Small data deduplication <http://dedupeio.github.io/dedupe-examples/docs/csv_example.html>`__
* `Record Linkage <https://dedupeio.github.io/dedupe-examples/docs/record_linkage_example.html>`__
* `Gazetter example <https://dedupeio.github.io/dedupe-examples/docs/gazetteer_example.html>`__
* `MySQL example <https://dedupeio.github.io/dedupe-examples/docs/mysql_example.html>`__
* `Postgres big dedupe example <https://dedupeio.github.io/dedupe-examples/docs/pgsql_big_dedupe_example.html>`__
* `Patent Author Disambiguation <https://dedupeio.github.io/dedupe-examples/docs/patent_example.html>`__

Errors / Bugs
=============

If something is not behaving intuitively, it is a bug, and should be
reported. `Report it here <https://github.com/dedupeio/dedupe/issues>`__

Contributing to dedupe
======================

Check out `dedupe <https://github.com/dedupeio/dedupe>`__
repo for how to contribute to the library.

Check out `dedupe-examples
<https://github.com/dedupeio/dedupe-examples>`__ for how to contribute
a useful example of using dedupe.

Citing dedupe
=============

If you use Dedupe in an academic work, please give this citation:

Gregg, Forest and Derek Eder. 2015. Dedupe. https://github.com/dedupeio/dedupe.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

