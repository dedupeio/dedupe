
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
