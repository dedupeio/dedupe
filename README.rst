Dedupe Python Library
=====================

Deduplication, entity resolution, record linkage, author disambiguation,
and others ...

As different research communities encountered this problem, they each
gave it a new name but, ultimately, its all about trying to figure out
what records are referring to the same thing.

**Dedupe is an open source python library that quickly de-duplicates
large sets of data.**

Features
^^^^^^^^

-  **machine learning** - reads in human labeled data to automatically
   create optimum weights and blocking rules
-  **runs on a laptop** - makes intelligent comparisons so you don't
   need a powerful server to run it
-  **built as a library** - so it can be integrated in to your
   applications or import scripts
-  **extensible** - supports adding custom data types, string
   comparators and blocking rules
-  **open source** - anyone can use, modify or add to it

How it works
^^^^^^^^^^^^

-  `Overview <https://github.com/datamade/dedupe/wiki/Home>`__
-  `Matching
   records <https://github.com/datamade/dedupe/wiki/Matching-records>`__
-  `Making smart
   comparisons <https://github.com/datamade/dedupe/wiki/Making-smart-comparisons>`__
-  `Grouping
   duplicates <https://github.com/datamade/dedupe/wiki/Grouping-duplicates>`__
-  `Choosing a good
   threshold <https://github.com/datamade/dedupe/wiki/Choosing-a-good-threshold>`__

Community
^^^^^^^^^

-  `Dedupe Google
   group <https://groups.google.com/forum/?fromgroups=#!forum/open-source-deduplication>`__
-  `ChiPy
   presentation <http://pyvideo.org/video/973/big-data-de-duping>`__
-  IRC channel, `#dedupe on
   irc.freenode.net <http://webchat.freenode.net/?channels=dedupe>`__

Installation and dependencies
-----------------------------

Dedupe requires `numpy <http://numpy.scipy.org/>`__, which can be
complicated to install. If you are installing numpy for the first time,
`follow these
instructions <http://docs.scipy.org/doc/numpy/user/install.html>`__.
You'll need to version 1.6 of numpy or higher.

After numpy is set up, then install the following: \*
`fastcluster <http://math.stanford.edu/~muellner/fastcluster.html>`__ \*
`hcluster <http://code.google.com/p/scipy-cluster/>`__ \*
`networkx <http://networkx.github.com/>`__ \*
`zope.index <https://pypi.python.org/pypi/zope.index>`__

Using pip:

.. code:: bash

    git clone git://github.com/datamade/dedupe.git
    cd dedupe
    pip install "numpy>=1.6"
    for python 2.7
    pip install -r requirements.txt
    # OR for python 2.6
    pip install -r py26_requirements.txt
    python setup.py install

Using easy\_install:

.. code:: bash

    git clone git://github.com/datamade/dedupe.git
    cd dedupe
    easy_install "numpy>=1.6"
    easy_install "fastcluster>=1.1.8"
    easy_install "hcluster>=0.2.0"
    easy_install networkx
    easy_install zope.interface
    easy_install zope.index
    python setup.py install

OS X Install Notes
~~~~~~~~~~~~~~~~~~

You may need to set the following environmental variables from the
command line
``bash export CFLAGS=-Qunused-arguments export CPPFLAGS=-Qunused-arguments``

With default configurations, dedupe cannot do parallel processing on Mac
OS X. For more information and for instructions on how to enable this,
`refer to the
wiki <https://github.com/datamade/dedupe/wiki/OSX-Install-Notes>`__.

Usage examples
--------------

Dedupe is a library and not a stand-alone command line tool. To
demonstrate its usage, we have come up with a few example recipes for
different sized datasets.

`CSV example <http://datamade.github.com/dedupe/doc/csv_example.html>`__ (<10,000 rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd examples/csv_example
    python csv_example.py

(use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f'
when you are finished)

**To see how you might use dedupe with smallish data, see the `annotated
source code for
csv\_example.py <http://datamade.github.com/dedupe/doc/csv_example.html>`__.**

`MySQL example <http://datamade.github.com/dedupe/doc/mysql_example.html>`__ (10,000 - 1,000,000+ rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can take a few hours and will noticeably tax your laptop. You might
want to run it overnight.

To follow this example you need to

-  Create a MySQL database called 'contributions'
-  Copy ``examples/mysql_example/mysql.cnf_LOCAL`` to
   ``examples/mysql_example/mysql.cnf``
-  Update ``examples/mysql_example/mysql.cnf`` with your MySQL username
   and password
-  ``easy_install MySQL-python`` or ``pip install MySQL-python``

Once that's all done you can run the example:

.. code:: bash

    cd examples/mysql_example
    python mysql_init_db.py 
    python mysql_example.py

(use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f'
when you are finished)

**To see how you might use dedupe with bigish data, see the `annotated
source code for
mysql\_example <http://datamade.github.com/dedupe/doc/mysql_example.html>`__.**

We are trying to figure out a range of typical runtimes for diferent
hardware. Please let us know your `run time for the MySQL
example <https://github.com/datamade/dedupe/wiki/Reported-MySQL-Example-Run-Times>`__.

`Record Linkage example <http://datamade.github.com/dedupe/doc/record_linkage_example.html>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example links two datasets, where each dataset, individually has no
duplicates.

.. code:: bash

    python examples/record_linkage_example/record_linkage_example.py 

**To see how you might use dedupe for linking datasets, see the
`annotated source code for
record\_linkage\_example.py <http://datamade.github.com/dedupe/doc/record_linkage_example.html>`__.**

Training
--------

The *secret sauce* of dedupe is human input. In order to figure out the
best rules to deduplicate a set of data, you must give it a set of
labeled examples to learn from.

The more labeled examples you give it, the better the deduplication
results will be. At minimum, you should try to provide **10 positive
matches** and **10 negative matches**.

The results of your training will be saved in a JSON file for future
runs of dedupe.

Here's an example labeling operation:

.. code:: bash

    Phone :  2850617
    Address :  3801 s. wabash
    Zip :
    Site name :  ada s. mckinley st. thomas cdc

    Phone :  2850617
    Address :  3801 s wabash ave
    Zip :
    Site name :  ada s. mckinley community services - mckinley - st. thomas

    Do these records refer to the same thing?
    (y)es / (n)o / (u)nsure / (f)inished

Documentation
-------------

`The documentation for the dedupe library is on our
wiki <https://github.com/datamade/dedupe/wiki/API-documentation>`__.

Testing
-------

` <https://travis-ci.org/datamade/dedupe>`__\ |Coverage Status|

Build extensions in place ``bash python setup.py build_ext --inplace``

Unit tests of core dedupe functions ``bash nosetests``

Test using canonical dataset from Bilenko's research
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Deduplication ``bash python tests/canonical_test.py``

Using Record Linkage ``bash python tests/canonical_test_matching.py``

Team
----

-  `Forest Gregg <mailto:fgregg@gmail.com>`__
-  `Derek Eder <mailto:derek.eder@gmail.com>`__

Credits
-------

Dedupe is based on Mikhail Yuryevich Bilenko's Ph.D. dissertation:
`*Learnable Similarity Functions and their Application to Record Linkage
and
Clustering* <http://www.cs.utexas.edu/~ml/papers/marlin-dissertation-06.pdf>`__.

Errors / Bugs
-------------

If something is not behaving intuitively, it is a bug, and should be
reported. `Report it here <https://github.com/datamade/dedupe/issues>`__

Note on Patches/Pull Requests
-----------------------------

-  Fork the project.
-  Make your feature addition or bug fix.
-  Send us a pull request. Bonus points for topic branches.

Copyright
---------

Copyright (c) 2013 Forest Gregg and Derek Eder. Released under the MIT
License.

`See LICENSE for
details <https://github.com/datamade/dedupe/wiki/License>`__

Third-party copyright in this distribution is noted where applicable.

Citing Dedupe
-------------

If you use Dedupe in an academic work, please give this citation:

Gregg, Forest, and Derek Eder. 2013. Dedupe.
https://github.com/datamade/dedupe.

|githalytics.com alpha|

.. |Coverage Status| image:: https://coveralls.io/repos/datamade/dedupe/badge.png?branch=master
   :target: https://coveralls.io/r/datamade/dedupe?branch=master
.. |githalytics.com alpha| image:: https://cruel-carlota.pagodabox.com/861a8f3ec74c8928e0baad77640ab042
   :target: http://githalytics.com/datamade/dedupe
