-  `API
   Documentation <https://github.com/datamade/dedupe/wiki/API-documentation>`__

Introduction to dedupe
======================

Problems with real-world data
-----------------------------

Journalists, academics, and businesses work hard to get big masses of
data to learn about what people or organizations are doing.
Unfortunately, once we get the data, we often can't answer our questions
because we can't tell who is who.

In much real-world data, we do not have a way of absolutely deciding
whether two records, say ``John Smith`` and ``J. Smith`` are referring
to the same person. If these were records of campaign contribution data,
did a ``John Smith`` give two donations or did ``John Smith`` and maybe
``Jane Smith`` give one contribution apiece?

People are pretty good at making these calls, if they have enough
information. For example, I would be pretty confident that the following
two records are the about the same person.

::

    first name | last name | address                   | phone   |
    --------------------------------------------------------------
    bob        | roberts   | 1600 pennsylvania ave.   | 555-0123 |
    Robert     | Roberts   | 1600 Pensylvannia Avenue |          |

If we have to decide which records in our data are about the same person
or organization, then we could just go through by hand, compare every
record, and decide which records are about the same entity.

This is very, very boring and can takes a **long** time. Dedupe is a
software library that can make these decisions about whether records are
about the same thing about as good as a person can, but quickly.

Blocking
--------

The first thing we do is define a way that a computer can calculate
whether two records are similar, and if they are similar whether they
are about the same thing. Unfortunately, even if we had the perfect way
to decide whether a pair of records are distinct or duplicates, there
are so many possible comparisons it would take years or millenia to
compute. So, next, we find a means to only compare records that we think
have a chance of being duplicates and avoid the great number of
fruitless comparisons of records that are very different.

Clustering
----------

Once we have decided whether pairs of records are duplicates, we have to
decide whether groups of three records or more are all duplicates. This
ends up being trickier than you might expect.

If all of the above steps are not perfect, and they won't be, we'll end
up saying some records are duplicates when they really are not and that
some records are not duplicates when they really are. We'll have to
decide which of these errors we care about more, and find a good way to
trade-off between them.

We get into more details on all of this below:

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   Matching-records
   Making-smart-comparisons
   Grouping-duplicates
   Choosing-a-good-threshold
   Special-Cases
   Bibliography
