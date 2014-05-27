.. py:method:: threshold(data_1, data_2, recall_weight)

   Returns the threshold that maximizes the expected F score, a weighted
   average of precision and recall for a sample of data.

   :param dict data_1: a dictionary of records from first dataset,
		       where the keys are record_ids and the
		       values are dictionaries with the keys being
		       field names.
   :param dict data_2: a dictionary of records from second dataset,
		       same form as data_1
   :param float recall_weight: sets the tradeoff between precision
			       and recall. I.e.  if you care twice
			       as much about recall as you do
			       precision, set recall_weight to 2.

   .. code:: python

       > threshold = deduper.threshold(data_1, data_2, recall_weight=2)
       > print threshold
       0.21

.. py:method:: match(data_1, data_2, threshold)

   Identifies pairs of records that refer to the same entity, returns
   tuples of record ids, where both record\_ids within a tuple should refer
   to the same entity

   This method should only used for small to moderately sized datasets for
   larger data, use matchBlocks

   :param dict data_1: a dictionary of records from first dataset,
		       where the keys are record_ids and the
		       values are dictionaries with the keys being
		       field names.
   :param dict data_2: a dictionary of records from second dataset,
		       same form as data_1
   :param float threshold: a number between 0 and 1 (default is
			   0.5). We will consider records as
			   potential duplicates if the predicted
			   probability of being a duplicate is
			   above the threshold.

			   Lowering the number will increase
			   recall, raising it will increase
			   precision
