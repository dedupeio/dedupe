===================
Grouping Duplicates
===================

Once we have calculated the probability that pairs of record are
duplicates or not, we still have a kind of thorny problem because it's
not just pairs of records that can be duplicates. Three, four, thousands
of records could all refer to the same entity (person, organization, ice
cream flavor, etc.,) but we only have pairwise measures.

Let's say we have measured the following pairwise probabilities between
records A, B, and C.

::

    A -- 0.6 -- B -- 0.6 -- C 

The probability that A and B are duplicates is 60%, the probability that
B and C are duplicates is 60%, but what is the probability that A and C
are duplicates?

Let's say that everything is going perfectly and we can say there's a
36% probability that A and C are duplicates. We'd probably want to say
that A and C should not be considered duplicates.

Okay, then should we say that A and B are a duplicate pair and C is a
distinct record or that A is the distinct record and that B and C are
duplicates?

Well... this is a thorny problem, and we tried solving it a few
different ways. In the end, we found that **hierarchical clustering with
centroid linkage** gave us the best results. What this algorithm does is
say that all points within some distance of centroid are part of the
same group. In this example, B would be the centroid - and A, B, C and
would all be put in the same group.

Unfortunately, a more principled answer does not exist because the
estimated pairwise probabilities are not transitive.

Clustering the groups depends on us setting a threshold for group
membership -- the distance of the points to the centroid. Depending on how
we choose that threshold, we'll get very different groups, and we will
want to choose this threshold wisely.

In recent years, there has been some very exciting research that
solves the problem of turning pairwise distances into clusters, by
avoiding making pairwise comparisons altogether. Unfortunately, these
developments are not compatible with Dedupe's pairwise approach. See,
`Michael Wick, et.al, 2012. "A Discriminative Hierarchical Model for Fast Coreference at Large Scale" <http://people.cs.umass.edu/~sameer/files/hierar-coref-acl12.pdf>`__
and `Rebecca C. Steorts, et. al., 2013. "A Bayesian Approach to Graphical Record Linkage and De-duplication" <http://arxiv.org/abs/1312.4645>`__.

