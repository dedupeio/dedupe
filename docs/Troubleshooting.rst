***************
Troubleshooting
***************

So you've tried to apply dedupe to your dataset, but you're having some problems.
Once you understand :ref:`how dedupe works <how-it-works-label>`, and you've taken
a look at some of the :doc:`examples<Examples>`, then this troubleshoooting
guide is your next step.

Memory Considerations
=====================

The top two likely memory bottlenecks, in order of likelihood, are:

1. Building the index predicates for blocking. If this is a problem,
   you can try turning off index blocking rules (and just use predicate
   blocking rules) by setting ``index_predicates=False`` in
   :meth:`dedupe.Dedupe.train`.

2. During `cluster()`. After scoring, we have to compare all the pairwise scores
   and build the clusters. dedupe runs a connected-components algorithm to
   determine where to begin the clustering, and this is currently done in
   memory using python dicts, so it can take substantial memory.
   There isn't currently a way to avoid this except to just use less records.

Time Considerations
===================

The slowest part of dedupe is probably during blocking. A big part of this is building
the index predicates, so the easiest fix for this is to set `index_predicates=False`
in :meth:`dedupe.Dedupe.train`.

Blocking could also be slow if dedupe has to do too many or too complex of
blocking rules. You can fix this by reducing the number of blocking rules dedupe has
to learn to cover all the true positives. Either you reduce the `recall` parameter
in :meth:`dedupe.Dedupe.train`, or, similarly, just use less positive examples
during training.

Note that you are making a choice here between speed and recall. The less blocking
you do, the faster you go, but the more likely you are to not block true positives
together.

This part of dedupe is still single-threaded, and could probably benefit
from parallelization or other code strategies,
although current attempts haven't really proved promising yet.


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

If the built in variables don't cut it, you can write your own variables.

Take a look at the separately maintained `optional variables
<https://github.com/search?q=org%3Adedupeio+dedupe-variable>`__
for examples of how to write your own custom variable types with
your custom comparators and predicates.