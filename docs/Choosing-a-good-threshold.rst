=========================
Choosing a Good Threshold
=========================

Dedupe can predict the *probability* that a pair of records are
duplicates. So, how should we decide that a pair of records really are
duplicates?

To answer this question we need to know something about Precision and
Recall. Why don't you check out the `Wikipedia
page <http://en.wikipedia.org/wiki/Precision_and_recall>`__ and come
back here.

There's always a trade-off between precision and recall. That's okay. As
long as we know how much we care about precision vs. recall, `we can
define an F-score <http://en.wikipedia.org/wiki/F1_score>`__ that will
let us find a threshold for deciding when records are duplicates *that
is optimal for our priorities*.

Typically, the way that we find that threshold is by looking at the true
precision and recall of some data where we know their true labels -
where we know the real duplicates. However, we will only get a good
threshold if the labeled examples are representative of the data we are
trying to classify.

So here's the problem - the labeled examples that we make with Dedupe
are not at all representative, and that's by design. In the active
learning step, we are not trying to find the most representative data
examples. We're trying to find the ones that will teach us the most.

The approach we take here is to take a random sample of blocked data,
and then calculate the pairwise probability that records will be
duplicates within each block. From these probabilities we can calculate
the expected number of duplicates and distinct pairs, so we can
calculate the expected precision and recall.

