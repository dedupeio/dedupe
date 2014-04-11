.. dedupe documentation master file, created by
   sphinx-quickstart on Thu Apr 10 11:27:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dedupe's documentation!
==================================

Contents:

.. toctree::
   :maxdepth: 1

   API-documentation  
   OSX-Install-Notes
   Home             

*Dedupe is an open source python library that quickly de-duplicates
large sets of data.*

Deduplication, entity resolution, record linkage, author
disambiguation, and others ...

As different research communities encountered this problem, they each
gave it a new name but, ultimately, its all about trying to figure out
what records are referring to the same thing.



Features
========
 * **machine learning** - reads in human labeled data to automatically create optimum weights and blocking rules
 * **runs on a laptop** - makes intelligent comparisons so you don't need a powerful server to run it
 * **built as a library** - so it can be integrated in to your applications or import scripts
 * **extensible** - supports adding custom data types, string comparators and blocking rules
 * **open source** - anyone can use, modify or add to it

Installation
============
.. code-block:: bash

   pip install "numpy>=1.6"
   pip install dedupe

Intro
=====

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




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

