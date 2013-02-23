# Dedupe Python Library
A free python library for accurate and scaleable deduplication and entity-resolution. 

[<img src="https://travis-ci.org/open-city/dedupe.png" />](https://travis-ci.org/open-city/dedupe)

Based on Mikhail Yuryevich Bilenko's Ph.D. dissertation: [*Learnable Similarity Functions and their Application to Record Linkage and Clustering*](http://www.cs.utexas.edu/~ml/papers/marlin-dissertation-06.pdf).

Current solutions break easily, don’t scale, and require significant developer time. Our solution is robust, can handle a large volume of data, and can be trained by anyone.

* For more detail and overview, [read the wiki](https://github.com/open-city/dedupe/wiki)
* [Join our Google group for updates](https://groups.google.com/forum/?fromgroups=#!forum/open-source-deduplication)
* [See our presentation at ChiPy](http://pyvideo.org/video/973/big-data-de-duping)

## Python Dependencies

This library requires [numpy](http://numpy.scipy.org/), which can be complicated to install. If you are installing numpy for the first time, [follow these instructions](http://docs.scipy.org/doc/numpy/user/install.html).

After numpy is set up, then install the following:
* [fastcluster](http://math.stanford.edu/~muellner/fastcluster.html)
* [hcluster](http://code.google.com/p/scipy-cluster/)
* [networkx](http://networkx.github.com/)

## Installation

Using pip:

```bash
pip install numpy
pip install -r requirements.txt
python setup.py install
```

Using easy_install:

```bash
easy_install numpy
easy_install fastcluster
easy_install hcluster
easy_install networkx
python setup.py install
```

## Usage examples

Dedupe is a library and not a stand-alone command line tool. To demonstrate its usage, we have come up with a few example recipes for different sized datasets.

### CSV example (<10,000 rows)
```bash
python examples/csv_example/csv_example.py
```
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished)
  
* [Annotated source code of csv_example](http://open-city.github.com/dedupe/doc/csv_example.html)
  
### sqlite example (10,000 - 1,000,000 rows)
```bash
python examples/sqlite_example/init_db.py
python examples/sqlite_example/sqlite_blocking.py
```
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished) 
  
```bash
python examples/sqlite_example/sqlite_clustering.py
```
* [Annotated source code of sqlite_blocking](http://open-city.github.com/dedupe/doc/sqlite_blocking.html)
* [Annotated source code of sqlite_clustering](http://open-city.github.com/dedupe/doc/sqlite_clustering.html)

*Please note*: We have been having performance with sqlite on some machines, particular in writing the 
blocking map table. If sqlite_blocking.py doesn't complete within eight hours, it probably will take days to
finish on your machine.

We are not sure if this is A. a problem with how we are using sqlite, B. a problem with using sqlite 
with this much data, C. a problem we will have with any database engine. We will implement a version
using MySQL soon to try to narrow down the problem. In the meantime, if you are an sqlite guru, we could
use your eyeballs.

### mysql example (10,000 - 1,000,000 rows)
To follow this example you need to 

* Create a MySQL database called 'contributions'
* Copy `examples/mysql_example/mysql.cnf_LOCAL` to `examples/mysql_example/mysql.cnf`
* Update `examples/mysql_example/mysql.cnf` with your MySQL username and password

Once that's all done you can run the example:

```bash
python examples/mysql_example/init_db.py
python examples/mysql_example/mysql_blocking.py
```
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished) 
  
```bash
python examples/mysql_example/mysql_clustering.py
```

## Testing

Unit tests of core dedupe functions
```bash
python test/test_dedupe.py
```

Test using canonical dataset from Bilenko's research
  
Using random sample data for training
```bash
python test/canonical_test.py
```

Using active learning for training
```bash
python test/canonical_test.py --active True
```

## Team

* [Forest Gregg](mailto:fgregg@gmail.com)
* [Derek Eder](mailto:derek.eder@gmail.com)

## Errors / Bugs

If something is not behaving intuitively, it is a bug, and should be reported.
[Report it here](https://github.com/open-city/dedupe/issues)


## Note on Patches/Pull Requests
 
* Fork the project.
* Make your feature addition or bug fix.
* Send us a pull request. Bonus points for topic branches.

## Copyright

Copyright (c) 2012 Forest Gregg and Derek Eder of Open City. Released under the MIT License.

[See LICENSE for details](https://github.com/open-city/dedupe/wiki/License)
