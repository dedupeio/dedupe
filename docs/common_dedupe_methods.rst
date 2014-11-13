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

.. py:method:: match(data, [threshold = 0.5, [max_components = 30000]])

   Identifies records that all refer to the same entity, returns
   tuples containing a sequence of record ids and corresponding
   sequence of confidence score as a float between 0 and 1. The
   record_ids within each set should refer to the same entity and the
   confidence score is a measure of our confidence a particular entity
   belongs in the cluster.
 
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
   :param int max_components: Dedupe splits records into connected
                              components and then clusters each
                              component. Clustering uses about N^2
                              memory, where N is the size of the
                              components.  Max components sets the
                              maximum size of a component dedupe will
                              try to cluster. If a component is larger
                              than max_components, dedupe will try to
                              split it into smaller
                              components. Defaults to 30K.

   .. code:: python

      > duplicates = deduper.match(data, threshold=0.5)
      > print duplicates
      [((1, 2, 3), 
        (0.790, 
         0.860, 
         0.790)), 
        ((4, 5), 
         (0.720, 
          0.720)), 
        ((10, 11), 
         (0.899, 
          0.899))]


.. py:method:: blocker(data)

   Generate the predicates for records. Yields tuples of (predicate,
   record_id).

   :param list data: A sequence of tuples of (record_id,
                     record_dict). Can often be created by
                     `data_dict.items()`.

   .. code:: python

      > data = [(1, {'name' : 'bob'}), (2, {'name' : 'suzanne'})]
      > blocked_ids = deduper.blocker(data)
      > print list(blocked_ids)
      [('foo:1', 1), ..., ('bar:1', 100)]
      


