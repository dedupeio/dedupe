.. py:method:: threshold(data[, recall_weight=1.5])

   Returns the threshold that maximizes the expected F score, a weighted
   average of precision and recall for a sample of data.

   :param dict data: a dictionary of records, where the keys are
		      record_ids and the values are dictionaries with
		      the keys being field names
   :param float recall_weight: sets the tradeoff between precision
				and recall. I.e.  if you care twice as
				much about recall as you do precision,
				set recall_weight to 2.


   .. code:: python

      > threshold = deduper.threshold(data, recall_weight=2)
      > print threshold
      0.21

.. py:method:: match(data, [threshold = 0.5])

   Identifies records that all refer to the same entity, returns tuples of
   record ids, where the record_ids within each tuple should refer to the
   same entity

   This method should only used for small to moderately sized datasets for
   larger data, use matchBlocks

   :param dict data: a dictionary of records, where the keys are
		      record_ids and the values are dictionaries with
		      the keys being field names
   :param float threshold: a number between 0 and 1 (default is 0.5).
			    We will consider records as potential
			    duplicates if the predicted probability of
			    being a duplicate is above the threshold.

			    Lowering the number will increase recall,
			    raising it will increase precision

   .. code:: python

      > duplicates = deduper.match(data, threshold=0.5)
      > print duplicates
      [(3,6,7), (2,10), ..., (11,14)]


.. py:method:: blocker(data)

   Generate the predicates for records. Yields tuples of (predicate,
   record_id).

   :param dict data: A dictionary-like object indexed by record ID
		      where the values are dictionaries representing records.

   .. code:: python

      > blocked_ids = deduper.blocker(data)
      > print list(blocked_ids)
      [('foo:1', 1), ..., ('bar:1', 100)]

