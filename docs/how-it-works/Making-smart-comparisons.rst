========================
Making Smart Comparisons
========================

Say we have magic function that takes in a pair of records and always
returns a ``False`` if a pair of records are distinct and ``True`` if a
pair of records refer to the same person or organization.

Let's say that this function was pretty slow. It always took one second
to return.

How long would it take to duplicate a thousand records?

Within a dataset of thousand records, there are :math:`\frac{1{,}000
\times 999}{2} = 499{,}500` unique pairs of records. If we
compared all of them using our magic function it would take six days.

But, one second is a **long** time, let's say we sped it up so that we
can make 10,000 comparisons per second. Now we can get through our
thousand-record-long dataset in less than a minute.

Feeling good about our super-fast comparison function, let's take on a
dataset of 100,000 records. Now there are
:math:`\frac{100{,}000 \times 99{,}999}{2} = 4{,}999{,}950{,}000` unique possible
pairs. If we compare all of them with our super-fast comparison function,
it will take six days again.

If we want to work with moderately sized data, we have to find a way of
making fewer comparisons.

Duplicates are rare
-------------------

In real world data, nearly all possible pairs of records are not
duplicates.

In this four-record example below, only two pairs of records are
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


If we could only compare records that were true duplicates, we wouldn't
run into the explosion of comparisons. Of course, if we already knew where
the true duplicates were, we wouldn't need to compare any individual
records. Unfortunately we don't, but we do quite well if just compare
records that are somewhat similar.

Blocking
--------

Duplicate records almost always share *something* in common. If we
define groups of data that share something and only compare the records
in that group, or *block*, then we can dramatically reduce the number of
comparisons we will make. If we define these blocks well, then we will make
very few comparisons and still have confidence that will compare records
that truly are duplicates.

This task is called blocking, and we approach it in two ways: predicate
blocks and index blocks.

Predicate blocks
~~~~~~~~~~~~~~~~

A predicate block is a bundle of records that all share a feature -- a
feature produced by a simple function called a predicate.

Predicate functions take in a record field, and output a set of features
for that field. These features could be "the first 3 characters of the
field," "every word in the field," and so on. Records that share the
same feature become part of a block.

Let's take an example. Let's use a "first 3 character" predicate on
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

Again, we're applying the "first three characters" predicate function to the
address field in our data, the function outputs the following features --
160, 160, 123, 123 -- and then we group together the records that have
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

.. _index-blocks-label:

Index Blocks
~~~~~~~~~~~~

Dedupe also uses another way of producing blocks from searching and
index. First, we create a special data structure, like an `inverted
index <http://en.wikipedia.org/wiki/Inverted_index>`__, that lets us
quickly find records similar to target records. We populate the index
with all the unique values that appear in field. 

When blocking, for each record we search the index for values similar to
the record's field. We block together records that share at least one
common search result.

Index predicates require building an index from all the unique values
in a field. This can take substantial time and memory. Index
predicates are also usually slower than predicate blocking.

Combining blocking rules
------------------------

If it's good to put define blocks of records that share the same 'city'
field, it might be even better to block records that share *both* the
'city' field *and* the 'zip code' field. Dedupe tries these cross-field
blocks. These combinations blocks are called disjunctive blocks.

Learning good blocking rules for given data
-------------------------------------------

Dedupe comes with a long set of predicates, and when these are
combined Dedupe can have hundreds of possible blocking rules to choose
from. We will want to find a small set of these rules that covers
every labeled duplicated pair but minimizes the total number pairs
dedupe will have to compare.

While we approach this problem by using greedy algorithms, particularly
`Chvatal's Greedy Set-Cover
algorithm <http://www.cs.ucr.edu/~neal/Papers/Young08SetCover.pdf>`__.

