# 2.0.6
- fixed bug that was preventing learning of index predicates in Dedupe mode

# 2.0.3
- Improved memory performance of connected components


# 2.0

- Python 3 only
- Static typing and type Hints
- Incorporate sqlite to extend normal API to millions of records
- Multiprocessing enabled for Windows
- Multiprocessing mode changed to spawn for Mac OS X
- Moved from CamelCase to lowercase_with_underscore for method names.
- Dropped ability to save indices in save settings.
- Moved from Deduper.match -> Dedupe.partition, RecordLink.match -> RecordLink.join, Gazetteer.match -> Gazetteer.search
- Renamed Matching.blocker -> Matching.fingerprinter
- Moved to autodoc for documentation
- Dropped threshold methods
- matchBlocks has been replaced by score, which takes pairs of records not blocks

# 1.10.0
- Dropped python 2.7 support

# 1.9.4
- Cleaned up block learning

# 1.9.3
- Improved performance of connected components algorithm with very large components
- Fixed pickling unpickling bug of Index predicate classes

# 1.9.0
- Implemented a disagreement based active labeler to improve blocking recall

# 1.8.2
- removed shelve-backed persistence in blocking data in favor of an improved in-memory implementation

# 1.8.0
- matchBlocks is not a generator; match is now optionally a generator. If the
  generator option is turned of for the Gazette match is lazy

# 1.7.8
- Speed up blocking, on our way to 3-predicates

# 1.7.5
- Significantly reduced memory footprint during connected_components

# 1.7.3
- Significantly reduced memory footprint during scoreDuplicates

# 1.7.2
- Improper release

# 1.7.1
- TempShelve class that addresses various bugs related to cleaning up tempoary shelves

# 1.7.0
- Added `target` argument to blocker and predicates for changing the behavior
  of the predicates for the target and source dataset if we are linking.

# 1.6.8
- Use file-backed blocking with dbm, dramatically increases size of data that can be handled without special programming

# 1.6.7
- Reduce memory footprint of matching

# 1.6.0
- Simplify .train method

# 1.5.5
- Levenshtein search based index predicates thanks to @mattandahalfew

# 1.5.0
- simplified the sample API, this might be a breaking change for some
- the active learner interface is now more modular to allow for a different learner
- random sampling of pairs has been improved for linking case and
  dedupe case, h/t to @MarkusShepherd

## 1.4.15
- frozendicts have finally been removed
- first N char predicates return their entire length if length is less
  than N, instead of nothing
- crossvalidation is skipped in active learning if using default rlr learner

## 1.4.5
- Block indexes can now be persisted by using the index=True argument
  in the writeSettings method

## 1.4.1
- Now uses C version of double metaphone for speed
- Much faster compounding of blocks in block learning

## 1.4.0
- Block learning now tries to minimize the total number of comparisons
  not just the comparisons of distinct records. This decouples makes
  block learning from learning classifier learning. This change has
  requires new, different arguments to the train method.

## 1.3.8
- Console labeler now shows fields in the order they are defined in
  the data model. The labeler also reports number of labeled examples
- `pud` argument added to the `train` method. Proportion of uncovered
  dupes. This deprecates `uncovered_dupes` argument

## 1.3.0
- If we have enough training data, consider Compound predicates of length 3 in addition to predicates of length 2

## 1.1.1
- None now treated as missing data indicator. Warnings for deprecations of older types of missing data indicators

## 1.1.0
Features
- Handle FuzzyCategoricalType in datamodel

## 1.0.0
Features
- Speed up learning
- Parallelize sampling
- Optional [CRF Edit Distance](https://dedupe.readthedocs.io/en/latest/Variable-definition.html#optional-edit-distance)

## 0.8.0
Support for Python 3.4 added. Support for Python 2.6 dropped.

Features
- Windows OS supported
- train method has argument for not considering index predicates
- TfIDFNGram Index Predicate added (for shorter string)
- SuffixArray Predicate
- Double Metaphone Predicates
- Predicates for numbers, OrderOfMagnitude, Round
- Set Predicate OrderOfCardinality
- Final, learned predicates list will now often be smaller without
  loss of coverage
- Variables refactored to support external extensions like
  https://github.com/datamade/dedupe-variable-address
- Categorical distance, regularized logistic regression, affine gap
  distance, canonicalization have been turned into separate libraries.
- Simplejson is now dependency

## 0.7.5
Features
- Individual record cluster membership scores
- New predicates
- New Exists Variable Type

Bug Fixes
- Latlong predicate fixed
- Set TFIDF canopy working properly

## 0.7.4
Features
- Sampling methods now use blocked sampling

## 0.7.0
Version 0.7.0 is backwards compatible, except for the match method of Gazetteer class

Features
- new index, unindex, and match methods in Gazetter Matching. Useful for
  streaming matching

## 0.6.0
Version 0.6.0 is *not* backwards compatible.

Features :
- new Text, ShortString, and exact string types
- multiple variables can be defined on same field
- new Gazette linker for matching dirty records against a master list
- performance improvements, particularly in memory usage
- canonicalize function in dedupe.convenience for creating a canonical representation of a cluster of records
- tons of bugfixes

API breaks
- when initializing an ActiveMatching object, `variable_definition` replaces `field_definition` and is a list of    dictionaries instead of a dictionary. See the documentation for details
- also when initializing a Matching object, `num_processes` has been replaced by `num_cores`, which now defaults to the
number of cpus on the machine
- when initializing a StaticMatching object, `settings_file` is now expected to be a file object not a string. The `readTraining`, `writeTraining`, `writeSettings` methods also all now expect file objects


## 0.5
Version 0.5 is *not* backwards compatible.

Features :

- Special case code for linking two datasets that, individually are unique
- Parallel processing using python standard library multiprocessing
- Much faster canopy creation using zope.index
- Asynchronous active learning methods

API breaks :
- `duplicateClusters` has been removed, it has been replaced by
  `match` and `matchBlocks`
- `goodThreshold` has been removed, it has been replaced by
  `threshold` and `thresholdBlocks`
- the meaning of `train` has changed. To train from training file use `readTraining`. To use console labeling, pass a dedupe instance to the `consoleLabel` function
- The convenience function dataSample has been removed. It has been replaced by
the `sample` methods
- It is no longer necessary to pass `frozendicts` to `Matching` classes
- `blockingFunction` has been removed and been replaced by the `blocker` method
