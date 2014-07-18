.. py:method:: match(messy_data, canonical_data, threshold=0.5, n_matches=1)

   Identifies pairs of records that could refer to the same entity,
   returns tuples containing tuples of possible matches, with a
   confidence score for each match. The record_ids within each tuple
   should refer to potential matches from a messy data record to
   canonical records. The confidence score is the estimated
   probability that the records refer to the same entity.

   This method should only used for small to moderately sized datasets for
   larger data, use matchBlocks

   :param dict messy_data: a dictionary of records from a messy
			   dataset, where the keys are record_ids and
			   the values are dictionaries with the keys
			   being field names.

   :param dict canonical_data: a dictionary of canonical records,
			       same form as messy_data
   :param float threshold: a number between 0 and 1 (default is
			   0.5). We will consider records as
			   potential duplicates if the predicted
			   probability of being a duplicate is
			   above the threshold.

			   Lowering the number will increase
			   recall, raising it will increase
			   precision
   :param int n_matches: the maximum number of possible matches from
			 canonical_data to return for each record in
			 messy_data. Defaults to 1


   .. code:: python
       > matches = gazetteer.match(messy_data, canonical_data, threshold=0.5, n_matches=2)
       > print matches
       [(((1, 6), 0.72), 
         ((1, 8), 0.6)), 
        (((2, 7), 0.72),), 
        (((3, 6), 0.72), 
         ((3, 8), 0.65)), 
        (((4, 6), 0.96), 
         ((4, 5), 0.63))]


.. py:method::  matchBlocks(blocks, threshold=.5, n_matches=2)

   Partitions blocked data and returns a list of clusters, where each
   cluster is a tuple of record ids

   .. code:: python

       clustered_dupes = deduper.matchBlocks(blocked_data, threshold)

   :param list blocks: Sequence of tuples of records, where each tuple
		       is a set of records covered by a blocking
		       predicate.

   :param float threshold: Number between 0 and 1 (default is .5). We
			   will only consider as duplicates record
			   pairs as duplicates if their estimated
			   duplicate likelihood is greater than the
			   threshold.

			   Lowering the number will increase recall,
			   raising it will increase precision.

   :param int n_matches: the maximum number of possible matches from
			 canonical_data to return for each record in
			 messy_data. Defaults to 1


