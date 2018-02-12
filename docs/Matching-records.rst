****************
Matching Records
****************

If you look at the following two records, you might think it's pretty
clear that they are about the same person.

::

    first name | last name | address                   | phone   |
    --------------------------------------------------------------
    bob        | roberts   | 1600 pennsylvania ave.   | 555-0123 |
    Robert     | Roberts   | 1600 Pensylvannia Avenue |          |

However, I bet it would be pretty hard for you to explicitly write down
all the reasons why you think these records are about the same Mr.
Roberts.

Record similarity
-----------------

One way that people have approached this problem is by saying that
records that are more similar are more likely to be duplicates. That's a
good first step, but then we have to precisely define what we mean for
two records to be similar.

The default way that we do this in Dedupe is to use what's called a
string metric. A string metric is an way of taking two strings and
returning a number that is low if the strings are similar and high if
they are dissimilar. One famous string metric is called the Hamming
distance. It counts the number of substitutions that must be made to
turn one string into another. For example, ``roberts`` and ``Roberts``
would have Hamming distance of 1 because we have to substitute ``r`` for
``R`` in order to turn ``roberts`` into ``Roberts``.

There are lots of different string metrics, and we actually use a metric
called the `Affine Gap Distance <https://en.wikipedia.org/wiki/Gap_penalty#Affine>`__, which is a
variation on the Hamming distance.

Record by record or field by field
----------------------------------

When we are calculating whether two records are similar we could treat
each record as if it was a long string.

::

    record_distance = string_distance('bob roberts 1600 pennsylvania ave. 555-0123',
                                      'Robert Roberts 1600 Pensylvannia Avenue')

Alternately, we could compare field by field

::

    record_distance = (string_distance('bob', 'Robert') 
                       + string_distance('roberts', 'Roberts')
                       + string_distance('1600 pennsylvania ave.', '1600 Pensylvannia Avenue')
                       + string_distance('555-0123', ''))

The major advantage of comparing field by field is that we don't have to
treat each field string distance equally. Maybe we think that its really
important that the last names and addresses are similar but it's not as
important that first name and phone numbers are close. We can express
that importance with numeric weights, i.e.

::

    record_distance = (0.5 * string_distance('bob', 'Robert') 
                       + 2.0 * string_distance('roberts', 'Roberts')
                       + 2.0 * string_distance('1600 pennsylvania ave.', '1600 Pensylvannia Avenue')
                       + 0.5 * string_distance('555-0123', ''))

Setting weights and making decisions
------------------------------------

Say we set our record\_distance to be this weighted sum of field
distances, just as we had above. Let's say we calculated the
record\_distance and we found that it was the beautiful number **8**.

That number, by itself, is not that helpful. Ultimately, we are trying
to decide whether a pair of records are duplicates, and I'm not sure
what decision I should make if I see an 8. Does an 8 mean that the pair
of records are really similar or really far apart, likely or unlikely to
be duplicates. We'd like to define the record distances so that we can
look at the number and know whether to decide whether it's a duplicate.

Also, I really would rather not have to set the weights by hand every
time. It can be very tricky to know which fields are going to matter and
even if I know that some fields are more important I'm not sure how to
quantify it (is it 2 times more important or 1.3 times)?

Fortunately, we can solve both problems with a technique called
regularized logistic regression. If we supply pairs of records that we
label as either being duplicates or distinct, then Dedupe will learn a
set of weights such that the record distance can easily be transformed
into our best estimate of the probability that a pair of records are
duplicates.

Once we have learned these good weights, we want to use them to find
which records are duplicates. But turns out that doing this the naive
way will usually not work, and :doc:`we'll have to do something
smarter <Making-smart-comparisons>`.

Active learning
~~~~~~~~~~~~~~~

In order to learn those weights, Dedupe needs example pairs with labels.
Most of the time, we will need people to supply those labels.

But the whole point of Dedupe is to save people's time, and that
includes making good use of your labeling time so we use an approach
called Active Learning.

Basically, Dedupe keeps track of bunch unlabeled pairs and whether

1. the current learning blocking rules would cover the pairs
2. the current learned classifier would predict that the pairs are
   duplicates or are distinct

We maintain a set of the pairs where there is disagreement: that is
pairs which classifier believes are duplicates but which are not
covered by the current blocking rules, and the pairs which the
classifier believes are distinct but which are blocked together.

Dedupe picks, at random from this disagreement set, a pair of records
and asks the user to decide. Once it gets this label, it relearns the
weights and blocking rules. We then recalculate the disagreement set.

Other field distances
~~~~~~~~~~~~~~~~~~~~~

We have implemented a number of field distance measures. See :doc:`the
details about variables <Variable-definition>`.


