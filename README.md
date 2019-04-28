# Dedupe Python Library
[![Linux build](https://img.shields.io/travis/dedupeio/dedupe.svg?style=flat-square&label=Linux%20build)](https://travis-ci.org/dedupeio/dedupe)[![Mac OS X build](https://img.shields.io/travis/dedupeio/dedupe.svg?style=flat-square&label=Mac%20OS%20X%20build)](https://travis-ci.org/dedupeio/dedupe)[![Windows build](https://img.shields.io/appveyor/ci/fgregg/dedupe-n4qju/master.svg?style=flat-square&label=Windows%20build)](https://ci.appveyor.com/project/fgregg/dedupe-n4qju)[![Coverage](https://img.shields.io/coveralls/dedupeio/dedupe.svg?style=flat-square)](https://coveralls.io/r/dedupeio/dedupe?branch=master)

_dedupe is a python library that uses machine learning to perform fuzzy matching, deduplication and entity resolution quickly on structured data._

__dedupe__ will help you: 

* __remove duplicate entries__ from a spreadsheet of names and addresses
* __link a list__ with customer information to another with order history, even without unique customer IDs
* take a database of campaign contributions and __figure out which ones were made by the same person__, even if the names were entered slightly differently for each record

dedupe takes in human training data and comes up with the best rules for your dataset to quickly and automatically find similar records, even with very large databases.

## Important links
* Documentation: https://docs.dedupe.io/
* Repository: https://github.com/dedupeio/dedupe
* Issues: https://github.com/dedupeio/dedupe/issues
* Mailing list: https://groups.google.com/forum/#!forum/open-source-deduplication
* Examples: https://github.com/dedupeio/dedupe-examples

## Tools built with dedupe

### [Dedupe.io](https://dedupe.io/)
A cloud service powered by the dedupe library for de-duplicating and finding matches in your data. It provides a step-by-step wizard for uploading your data, setting up a model, training, clustering and reviewing the results.

[Dedupe.io](https://dedupe.io/) also supports record linkage across data sources and continuous matching and training through an [API](https://apidocs.dedupe.io/en/latest/).

For more, see the [Dedupe.io product site](https://dedupe.io/), [tutorials on how to use it](https://dedupe.io/tutorial/intro-to-dedupe-io.html), and [differences between it and the dedupe library](https://dedupe.io/documentation/should-i-use-dedupeio-or-the-dedupe-python-library.html).


### [csvdedupe](https://github.com/dedupeio/csvdedupe)
Command line tool for de-duplicating and [linking](https://github.com/dedupeio/csvdedupe#csvlink-usage) CSV files. Read about it on [Source Knight-Mozilla OpenNews](https://source.opennews.org/en-US/articles/introducing-cvsdedupe/).

## Installation

### Using dedupe

If you only want to use dedupe, install it this way:

```bash
pip install "numpy>=1.9"
pip install dedupe
```

Familiarize yourself with [dedupe's API](https://docs.dedupe.io/en/latest/API-documentation.html), and get started on your project. Need inspiration? Have a look at [some examples](https://github.com/dedupeio/dedupe-examples).

### Developing dedupe

We recommend using [virtualenv](http://virtualenv.readthedocs.org/en/latest/virtualenv.html) and [virtualenvwrapper](http://virtualenvwrapper.readthedocs.org/en/latest/install.html) for working in a virtualized development environment. [Read how to set up virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

Once you have virtualenvwrapper set up,

```bash
mkvirtualenv dedupe
git clone git://github.com/dedupeio/dedupe.git
cd dedupe
pip install "numpy>=1.9"
pip install -r requirements.txt
cython src/*.pyx
pip install -e .
```

If these tests pass, then everything should have been installed correctly!

```bash
pytest
```

Afterwards, whenever you want to work on dedupe,

```bash
workon dedupe
```

## Testing
Unit tests of core dedupe functions
```bash
pytest
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

* Forest Gregg, DataMade
* Derek Eder, DataMade

## Credits

Dedupe is based on Mikhail Yuryevich Bilenko's Ph.D. dissertation: [*Learnable Similarity Functions and their Application to Record Linkage and Clustering*](http://www.cs.utexas.edu/~ml/papers/marlin-dissertation-06.pdf).

## Errors / Bugs

If something is not behaving intuitively, it is a bug, and should be reported.
[Report it here](https://github.com/dedupeio/dedupe/issues)


## Note on Patches/Pull Requests
 
* Fork the project.
* Make your feature addition or bug fix.
* Send us a pull request. Bonus points for topic branches.

## Copyright

Copyright (c) 2019 Forest Gregg and Derek Eder. Released under the [MIT License](https://github.com/dedupeio/dedupe/blob/master/LICENSE).

Third-party copyright in this distribution is noted where applicable.

## Citing Dedupe
If you use Dedupe in an academic work, please give this citation:

Forest Gregg and Derek Eder. 2019. Dedupe. https://github.com/dedupeio/dedupe.
