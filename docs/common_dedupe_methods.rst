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

.. py:attribute:: blocker.index_fields 
   
   A dictionary of the Index Predicates that will used for blocking. The
   keys are the fields the predicates will operate on. 

.. py:method:: blocker.index(field_data, field)

   Indexes the data from a field for use in a index predicate. 

   :param set field data: The unique field values that appear in your data.
   :param string field: The name of the field

   .. code:: python

      for field in deduper.blocker.index_fields :
	     field_data = set(record[field] for record in data)
	     deduper.index(field_data, field)


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


.. py:method::  matchBlocks(blocks, [threshold=.5])

   Partitions blocked data and returns a list of clusters, where each
   cluster is a tuple of record ids

   .. code:: python

   Keyword arguments

   :param list blocks: Sequence of records blocks. Each record block
		       is a tuple containing records to compare. Each
		       block should contain two or more records.
		       Along with each record, there should also be
		       information on the blocks that cover that
		       record.

		       For example, if we have three records: 

		       .. code :: python
		           
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Pat', 'address' : '123 Main'})
			  (3, {'name' : 'Sam', 'address' : '123 Main'})

		       and two predicates: "Whole name" and "Whole address".
		       These predicates will produce the following blocks:

		       .. code :: python

		          # Block 1 (Whole name)
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Pat', 'address' : '123 Main'})

			  # Block 2 (Whole name)
			  (3, {'name' : 'Sam', 'address' : '123 Main'})

			  # Block 3 (Whole address
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Pat', 'address' : '123 Main'})
			  (3, {'name' : 'Sam', 'address' : '123 Main'})

		       So, the blocks you feed to matchBlocks should look
		       like this, after filtering out the singleton block.

		       .. code :: python

		          blocks =((
			            (1, {'name' : 'Pat', 'address' : '123 Main'}, set([])),
			            (2, {'name' : 'Pat', 'address' : '123 Main'}, set([]))
				    ), 
			           (
				    (1, {'name' : 'Pat', 'address' : '123 Main'}, set([1])),
			            (2, {'name' : 'Pat', 'address' : '123 Main'}, set([1])),
			            (3, {'name' : 'Sam', 'address' : '123 Main'}, set([]))
				    )
				   )
			  deduper.matchBlocks(blocks)

		       Within each block, dedupe will compare every
		       pair of records. This is expensive. Checking to
		       see if two sets intersect is much cheaper, and
		       if the block coverage information for two
		       records does intersect, that means that this
		       pair of records has been compared in a previous
		       block, and dedupe will skip comparing this pair
		       of records again.

   :param float threshold: Number between 0 and 1 (default is .5). We
			   will only consider as duplicates record
			   pairs as duplicates if their estimated
			   duplicate likelihood is greater than the
			   threshold.

			   Lowering the number will increase recall,
			   raising it will increase precision.
      


