========================
Making Smart Comparisons
========================

Say we have magic function that takes in a pair of records and always
returns a ``False`` if a pair of records are distinct and ``True`` if a
pair of records refer to the same person or organization.

Let's say that this function was pretty slow. It always took one second
to return.

How long would it take to duplicate a thousand records?

Within a data set of thousand records, there are :math:`\frac{1{,}000
\times 999}{2} = 499{,}500` unique pairs of records. If we
compared all of them using our magic function it would take six days.

But, one second is a **long** time, let's say we sped it up so that we
can make 10,000 comparisons per second. Now we can get through our
thousand record long dataset in less than a minute

Feeling good about our super fast comparison function, let's take on a
data set of 100,000 records. Now there are
:math:`\frac{100{,}000 \times 99{,}999}{2} = 4{,}999{,}950{,}000` unique possible
pairs. If we compare all of them with our super fast comparison function,
it will take six days again.

If we want to work with moderately sized data, we have to find a way of
making fewer comparisons.

Duplicates are rare
-------------------

In real world data, nearly all possible pairs of records are not
duplicates.

In this four record example below, only two pairs of records are
duplicates--(1, 2) and (3, 4), while there are four unique
pairs of records that are not duplicates--(1,3), (1,4), (2,3), and (2,4). 
Typically, as the size of the dataset grows, the fraction of pairs of records
that are duplicates gets very small very quickly.

+-------------+-----------+--------------------------+--------------+----------+
|  first name | last name | address                  | phone        | record_id|
+=============+===========+==========================+==============+==========+
|  bob        | roberts   | 1600 pennsylvania ave.   | 555-0123     | 1        |
+-------------+-----------+--------------------------+--------------+----------+
|  Robert     | Roberts   | 1600 Pensylvannia Avenue |              | 2        |
+-------------+-----------+--------------------------+--------------+----------+
|  steve      | Jones     | 123 Cowabunga Lane       | 555-0000     | 3        |
+-------------+-----------+--------------------------+--------------+----------+
|  Stephen    | Janes     | 123 Cawabunga Ln         | 444-555-0000 | 4        |
+-------------+-----------+--------------------------+--------------+----------+


If we could only compare records that were true duplicates we would not
run into the explosion of comparisons. Of course if already knew where
the true duplicates were, we wouldn't need to compare any individual
records. Unfortunately we don't but we do quite well if just compare
records that are somewhat similar.

Blocking
--------

Duplicate records almost always share some *thing* in common. If we
define groups of data that share some thing and only compare the records
in that group, or *block*, then we can dramatically reduce the number of
comparisons we will make. If define these blocks well, then we will make
very few comparisons and still have confidence that will compare records
that truly are duplicates.

This task is called blocking, and we approach it in two ways: predicate
blocks and canopies.

Predicate blocks
~~~~~~~~~~~~~~~~

A predicate block is a bundle of records that all share a feature - a
feature produced by a simple function called a predicate.

Predicate functions take in a record field, and output a set of features
for that field. These features could be "the first 3 characters of the
field," "every word in the field," and so on. Records that share the
same feature become part of a block.

Let's take an example. Let's use use a "first 3 character" predicate on
the **address field** below..

+-------------+-----------+--------------------------+--------------+----------+
|  first name | last name | address                  | phone        | record_id|
+=============+===========+==========================+==============+==========+
|  bob        | roberts   | 1600 pennsylvania ave.   | 555-0123     | 1        |
+-------------+-----------+--------------------------+--------------+----------+
|  Robert     | Roberts   | 1600 Pensylvannia Avenue |              | 2        |
+-------------+-----------+--------------------------+--------------+----------+
|  steve      | Jones     | 123 Cowabunga Lane       | 555-0000     | 3        |
+-------------+-----------+--------------------------+--------------+----------+
|  Stephen    | Janes     | 123 Cawabunga Ln         | 444-555-0000 | 4        |
+-------------+-----------+--------------------------+--------------+----------+

That leaves us with two blocks - The '160' block, which contains records
1 and 2, and the '123' block, which contains records 3 and 4.

::

    {'160' : (1,2) # tuple of record_ids
     '123' : (3,4)
     } 

Again, we're applying the "first three words" predicate function to the
address field in our data, the function outputs the following features -
160, 160, 123, 123 - and then we group together the records that have
identical features into "blocks".

Others simple predicates Dedupe uses include: 

* whole field 
* token field 
* common integer 
* same three char start 
* same five char start
* same seven char start 
* near integers 
* common four gram 
* common six gram

Canopies
~~~~~~~~

Dedupe also uses another way of producing blocks called canopies, which
`was developed by Andrew McCallum, Kamal Nigamy, and Lyle
Ungar <http://www.kamalnigam.com/papers/canopy-kdd00.pdf>`__.

Here's the basic idea: We start with the first record in our data. We
add it to a new canopy group and call it the target record. We then go
through the rest of the records. If a record is close enough to target
record then we add it to the canopy group. Once we have passed through
the data, we find the next record that was not assigned to a canopy and
this become our target record for a new canopy and we repeat the process
until every record is part of a canopy.

In order to build canopies, we need a distance that we can compute
without have to compare every single record, which is what we were
trying to avoid in the first place. We use the the `cosine of TF/IDF
weighted word
vectors <http://en.wikipedia.org/wiki/Vector_Space_Model>`__.

Combining blocking rules
------------------------

If it's good to put define blocks of records that share the same 'city'
field, it might be even better to block record that share BOTH the
'city' field AND 'zip code' field. Dedupe tries these cross-field
blocks. These combinations blocks are called disjunctive blocks.

Learning good blocking rules for given data
-------------------------------------------

Dedupe comes with a long set of predicate blocks and can create canopies
for any field. When these are combined dedupe have hundreds of possible
blocking rules to choose from. We will want to find a small set of these
rules that minimizes the number of distinct records in a block while
ensuring that nearly all true duplicates are in some block.

While we approach this problem by using greedy algorithm, particularly
`Chvatal's Greedy Set-Cover
algorithm <http://www.cs.ucr.edu/~neal/Papers/Young08SetCover.pdf>`__.
With a set of pairs that are labeled as distinct pairs or duplicate
pairs, and we try to find the best set of predicates.

