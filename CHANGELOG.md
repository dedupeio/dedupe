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
