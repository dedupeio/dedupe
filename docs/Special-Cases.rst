=============
Special Cases
=============

The process we have been describing is for the most general case--when
you have a dataset where an arbitrary number of records can all refer to
the same entity.

There are certain special cases where we can make more assumptions about
how records can be linked, which if true, make the problem much simpler.

One important case we call Record Linkage. Say you have two datasets and
you want to find the records in each dataset that refer to the same
thing. If you can assume that each dataset, individually, is unique,
then this puts a big constraint on how records can match. If this
uniqueness assumption holds, then (A) two records can only refer to the
same entity if they are from different datasets and (B) no other record
can match either of those two records.

