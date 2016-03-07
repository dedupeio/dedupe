# Dedupe Python Library
[![Linux build](https://img.shields.io/travis/datamade/dedupe.svg?style=flat-square&label=Linux build)](https://travis-ci.org/datamade/dedupe)[![Windows build](https://img.shields.io/appveyor/ci/fgregg/dedupe.svg?style=flat-square&label=Windows build)](https://ci.appveyor.com/project/fgregg/dedupe)[![Coverage](https://img.shields.io/coveralls/datamade/dedupe.svg?style=flat-square)](https://coveralls.io/r/datamade/dedupe?branch=master)

_dedupe is a python library that uses machine learning to perform de-duplication and entity resolution quickly on structured data._

__dedupe__ will help you: 

* __remove duplicate entries__ from a spreadsheet of names and addresses
* __link a list__ with customer information to another with order history, even without unique customer id's
* take a database of campaign contributions and __figure out which ones were made by the same person__, even if the names were entered slightly differently for each record

dedupe takes in human training data and comes up with the best rules for your dataset to quickly and automatically find similar records, even with very large databases.

## Important links
* Documentation: http://dedupe.rtfd.org/
* Repository: https://github.com/datamade/dedupe
* Issues: https://github.com/datamade/dedupe/issues
* Examples: https://github.com/datamade/dedupe-examples
* IRC channel, [#dedupe on irc.freenode.net](http://webchat.freenode.net/?channels=dedupe)

## Tools built with dedupe

### [csvdedupe](https://github.com/datamade/csvdedupe)
Command line tool for de-duplicating and [linking](https://github.com/datamade/csvdedupe#csvlink-usage) CSV files. Read about it on [Source Knight-Mozilla OpenNews](https://source.opennews.org/en-US/articles/introducing-cvsdedupe/).

## Installation

### Users

If you only want to use dedupe, install it this way:

```bash
pip install "numpy>=1.9"
pip install dedupe
```

### Developers

```bash
git clone git://github.com/datamade/dedupe.git
cd dedupe
pip install "numpy>=1.9"
pip install -r requirements.txt
cython src/*.pyx
pip install -e .

#If these tests pass, then everything should have been installed correctly!
nosetests
```

## Testing
Unit tests of core dedupe functions
```bash
nosetests
```

#### Test using canonical dataset from Bilenko's research
  
Using Deduplication
```bash
python tests/canonical.py
```

Using Record Linkage
```bash
python tests/canonical_matching.py
```


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

Copyright (c) 2016 Forest Gregg and Derek Eder. Released under the [MIT License](https://github.com/datamade/dedupe/blob/master/LICENSE).

Third-party copyright in this distribution is noted where applicable.

## Citing Dedupe
If you use Dedupe in an academic work, please give this citation:

Gregg, Forest, and Derek Eder. 2014. Dedupe. https://github.com/datamade/dedupe.
