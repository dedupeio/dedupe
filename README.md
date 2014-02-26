# Dedupe Python Library

Deduplication, entity resolution, record linkage, author disambiguation, and others ...

As different research communities encountered this problem, they each gave it a new name but, ultimately, its all about trying to figure out what records are referring to the same thing.

__Dedupe is an open source python library that quickly de-duplicates large sets of data.__

#### Features
 * __machine learning__ - reads in human labeled data to automatically create optimum weights and blocking rules
 * __runs on a laptop__ - makes intelligent comparisons so you don't need a powerful server to run it
 * __built as a library__ - so it can be integrated in to your applications or import scripts
 * __extensible__ - supports adding custom data types, string comparators and blocking rules
 * __open source__ - anyone can use, modify or add to it

#### How it works
* [Overview](https://github.com/datamade/dedupe/wiki/Home)
* [Matching records](https://github.com/datamade/dedupe/wiki/Matching-records)
* [Making smart comparisons](https://github.com/datamade/dedupe/wiki/Making-smart-comparisons)
* [Grouping duplicates](https://github.com/datamade/dedupe/wiki/Grouping-duplicates)
* [Choosing a good threshold](https://github.com/datamade/dedupe/wiki/Choosing-a-good-threshold)

#### Community
* [Dedupe Google group](https://groups.google.com/forum/?fromgroups=#!forum/open-source-deduplication)
* [ChiPy presentation](http://pyvideo.org/video/973/big-data-de-duping)
* IRC channel, #dedupe on irc.freenode.net

## Installation and dependencies


Dedupe requires [numpy](http://numpy.scipy.org/), which can be complicated to install. 
If you are installing numpy for the first time, 
[follow these instructions](http://docs.scipy.org/doc/numpy/user/install.html). You'll need to version 1.6 of numpy or higher.

After numpy is set up, then install the following:
* [fastcluster](http://math.stanford.edu/~muellner/fastcluster.html)
* [hcluster](http://code.google.com/p/scipy-cluster/)
* [networkx](http://networkx.github.com/)
* [zope.index](https://pypi.python.org/pypi/zope.index)

Using pip:

```bash
git clone git://github.com/datamade/dedupe.git
cd dedupe
pip install "numpy>=1.6"
# for python 2.7
pip install -r requirements.txt
# OR for python 2.6
pip install -r py26_requirements.txt
python setup.py install
```

Using easy_install:

```bash
git clone git://github.com/datamade/dedupe.git
cd dedupe
easy_install "numpy>=1.6"
easy_install "fastcluster>=1.1.8"
easy_install "hcluster>=0.2.0"
easy_install networkx
easy_install zope.interface
easy_install zope.index
python setup.py install
```

## Usage examples

Dedupe is a library and not a stand-alone command line tool. To demonstrate its usage, we have come up with a few example recipes for different sized datasets.

### [CSV example](http://datamade.github.com/dedupe/doc/csv_example.html) (<10,000 rows)
```bash
cd examples/csv_example
python csv_example.py
```
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished)
  
**To see how you might use dedupe with smallish data, see the [annotated source code for csv_example.py](http://datamade.github.com/dedupe/doc/csv_example.html).**

### [MySQL example](http://datamade.github.com/dedupe/doc/mysql_example.html) (10,000 - 1,000,000+ rows)
This can take a few hours and will noticeably tax your laptop. You might want to run it overnight.

To follow this example you need to 

* Create a MySQL database called 'contributions'
* Copy `examples/mysql_example/mysql.cnf_LOCAL` to `examples/mysql_example/mysql.cnf`
* Update `examples/mysql_example/mysql.cnf` with your MySQL username and password
* `easy_install MySQL-python` or `pip install MySQL-python`

Once that's all done you can run the example:

```bash
cd examples/mysql_example
python mysql_init_db.py 
python mysql_example.py
```
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished) 

**To see how you might use dedupe with bigish data, see the [annotated source code for mysql_example](http://datamade.github.com/dedupe/doc/mysql_example.html).** 

We are trying to figure out a range of typical runtimes for diferent hardware. Please let us know your 
[run time for the MySQL example](https://github.com/datamade/dedupe/wiki/Reported-MySQL-Example-Run-Times).

### [Record Linkage example](http://datamade.github.com/dedupe/doc/record_linkage_example.html) 
This example links two datasets, where each dataset, individually has no duplicates.

```bash
python examples/record_linkage_example/record_linkage_example.py 
```

**To see how you might use dedupe for linking datasets, see the [annotated source code for record_linkage_example.py](http://datamade.github.com/dedupe/doc/record_linkage_example.html).**


## Documentation
[The documentation for the dedupe library is on our wiki](https://github.com/datamade/dedupe/wiki/API-documentation).

## Testing

[<img src="https://travis-ci.org/datamade/dedupe.png" />](https://travis-ci.org/datamade/dedupe)[![Coverage Status](https://coveralls.io/repos/datamade/dedupe/badge.png?branch=master)](https://coveralls.io/r/datamade/dedupe?branch=master)

Build extensions in place
```bash
python setup.py build_ext --inplace
```

Unit tests of core dedupe functions
```bash
nosetests
```

#### Test using canonical dataset from Bilenko's research
  
Using Deduplication
```bash
python tests/canonical_test.py
```

Using Record Linkage
```bash
python tests/canonical_test_matching.py
```

## OS X Install Notes

With default settings, dedupe cannot do parallel processing on Mac OS X. According to their documentation, NumPy does not actually require Linear Algebra libraries (such as OpenBLAS) to be installed. To compile NumPy without BLAS support:

``` bash
$ export BLAS=None
$ pip install numpy
```

#### Alternatively, clone and build OpenBLAS

If you want to enable parallel processing you need to install some additional software libraries. One way to get around this is to compile NumPy against a different implementation of BLAS such as [OpenBLAS](https://github.com/xianyi/OpenBLAS). Here’s how you might go about that

``` bash 
$ git clone https://github.com/xianyi/OpenBLAS.git 
$ cd OpenBLAS 
$ make USE_OPENMP=0 # This compile flag is key
$ mkdir /usr/local/opt/openblas # Change this to suit your needs 
$ make PREFIX=/usr/local/opt/openblas install # Make sure this matches the path above 
```

#### Clone and build NumPy 

Make sure it knows where you just built OpenBLAS. This involves editing the site.cfg file within the NumPy source (see http://stackoverflow.com/a/14391693/1907889 for details). The paths that you’ll enter in there are relative to the ones used in step one above.

## Team

* [Forest Gregg](mailto:fgregg@gmail.com)
* [Derek Eder](mailto:derek.eder@gmail.com)

## Credits

Dedupe is based on Mikhail Yuryevich Bilenko's Ph.D. dissertation: [*Learnable Similarity Functions and their Application to Record Linkage and Clustering*](http://www.cs.utexas.edu/~ml/papers/marlin-dissertation-06.pdf).

## Errors / Bugs

If something is not behaving intuitively, it is a bug, and should be reported.
[Report it here](https://github.com/datamade/dedupe/issues)


## Note on Patches/Pull Requests
 
* Fork the project.
* Make your feature addition or bug fix.
* Send us a pull request. Bonus points for topic branches.

## Copyright

Copyright (c) 2013 Forest Gregg and Derek Eder. Released under the MIT License.

[See LICENSE for details](https://github.com/datamade/dedupe/wiki/License)

Third-party copyright in this distribution is noted where applicable.

## Citing Dedupe
If you use Dedupe in an academic work, please give this citation:

Gregg, Forest, and Derek Eder. 2013. Dedupe. https://github.com/datamade/dedupe.

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/861a8f3ec74c8928e0baad77640ab042 "githalytics.com")](http://githalytics.com/datamade/dedupe)
