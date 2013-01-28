# Dedupe Python Library
A free python library for accurate and scaleable deduplication and entity-resolution. 

[<img src="https://travis-ci.org/open-city/dedupe.png" />](https://travis-ci.org/open-city/dedupe)

Based on Mikhail Yuryevich Bilenko's Ph. D dissertation: Learnable Similarity Functions and their Application to Record Linkage and Clustering

Current solutions break easily, don’t scale, and require significant developer time. Our solution is robust, can handle a large volume of data, and can be trained by anyone.

* For more detail and overview, [read the wiki](https://github.com/open-city/dedupe/wiki)
* [Join our Google group for updates](https://groups.google.com/forum/?fromgroups=#!forum/open-source-deduplication)
* [See our presentation at ChiPy](http://pyvideo.org/video/973/big-data-de-duping)

## Python Dependencies

* [numpy](http://numpy.scipy.org/)
* [fastcluster](http://math.stanford.edu/~muellner/fastcluster.html)
* [hcluster](http://code.google.com/p/scipy-cluster/)
* [networkx](http://networkx.github.com/)

## Usage
  > python setup.py install
  > python examples/csv_example.py
  (use 'y', 'n' and 'u' keys to flag duplicates for active learning, 'f' when you are finished) 

## Testing

Unit tests of core dedupe functions
  > python tests/test_dedupe.py

Test using canonical dataset from Bilenko's research
  
Using random sample data for training
  > python tests/canonical_test.py

Using active learning for training
  > python tests/canonical_test.py --active True

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
