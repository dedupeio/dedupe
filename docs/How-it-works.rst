############
How it works
############

.. toctree::
   :maxdepth: 1

   Matching-records
   Making-smart-comparisons
   Grouping-duplicates
   Choosing-a-good-threshold
   Special-Cases


**Problems with real-world data**

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
