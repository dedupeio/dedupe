**********
User Guide
**********

Once you understand :ref:`how dedupe works <how-it-works-label>`, and you've taken
a look at some of the :doc:`examples<Examples>`,
you probably want to figure out how to adapt dedupe to your own problem. This
user guide is for that.

Memory Considerations
=====================

The most common memory bottleneck is during clustering. Within every block,
after computing a score for each pair of records, dedupe then runs a
connected-components algorithm as a pre-processing step for the hierarchical
agglomerative clustering.
`Currently, this requires having the entire edge list in memory
<https://github.com/dedupeio/dedupe/issues/819>`__, which for :math:`N`` records
leads to :math:`O(N^2)` memory usage. If you end up with any "superblocks" because
your blocking rules weren't picky enough, this can run you out of memory. You
could inspect your blocks prior to clustering, and possibly split them up further.

The other major memory bottleneck is building the index for blocking. If this is
a problem, you can try turning off index blocking rules (and just use predicate
blocking rules) by setting ``index_predicates=False`` in
:meth:`dedupe.Dedupe.train`.

Time Considerations
===================

Similar to the memory considerations, the slow part of dedupe is whenever
we have to run an algorithm that has a time complexity :math:`> O(N)`. The
scoring of pairs ():math:`O(N^2)`) is already implemented in a parallelized way,
but the clustering is not (yet?), so again this clustering step is probably
going to be your bottleneck. Again, inspecting and splitting up superblocks
can help.


Improving Accuracy
==================

- Inspect your results and see if you can find any patterns: Does dedupe
  not seem to be paying enough attention to some detail?

- Inspect the pairs given to you during :func:`dedupe.console_label`. These
  are pairs that dedupe is most confused about. Are these actually confusing
  pairs? If so, then great, dedupe is doing about as well as you could expect.
  If the pair is obviously a duplicate or obviously not a duplicate, then this
  means there is some signal that you should help dedupe to find.

- Read up on the theory behind each of the variable types. Some of them
  are going to work better depending on the situation, so try to understand
  them as well as you can.

- Add other variables. For instance try treating a field as both a `String`
  and as a `Text` variable. If this doesn't cut it, add your own custom
  variable that emphasizes the feature that you're really looking for.
  For instance, if you have a list of last names, you might want "Smith"
  to score well with "Smith-Johnson" (someone got married?). None of the
  builtin variables will handle this well, so write your own comparator.

- Add `Interaction` variables. For instance, if both the "last name" and 
  "street address" fields score very well, then this is almost a guarantee
  that these two records refer to the same person. An `Interaction` variable
  can emphasize this to the learner.

Extending Dedupe
================

Take a look at the separately maintained `optional variables
<https://github.com/search?q=org%3Adedupeio+dedupe-variable>`__
for examples of how to write your own custom variable types with
your custom comparators and predicates.