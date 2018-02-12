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

.. py:method:: match(data_1, data_2[, threshold=0.5[, generator=False]])

   Identifies pairs of records that refer to the same entity, returns tuples
   containing a set of record ids and a confidence score as a float between 0
   and 1. The record_ids within each set should refer to the
   same entity and the confidence score is the estimated probability that 
   the records refer to the same entity.

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
   :param bool generator: when `True`, match will generate a sequence
                          of clusters, instead of a list. Defaults to `False`

.. py:method::  matchBlocks(blocks, [threshold=.5])

   Partitions blocked data and returns a list of clusters, where each
   cluster is a tuple of record ids

   .. code:: python

   Keyword arguments

   :param list blocks: Sequence of records blocks. Each record block
		       is a tuple containing two sequences of records,
		       the records from the first data set and the
		       records from the second dataset. Within each
		       block there should be at least one record from
		       each datasets.  Along with each record, there
		       should also be information on the blocks that
		       cover that record.

		       For example, if we have two records from dataset
		       A and one record from dataset B: 

		       .. code :: python
		           
		          # Dataset A
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Sam', 'address' : '123 Main'})

			  # Dataset B
			  (3, {'name' : 'Pat', 'address' : '123 Main'})

		       and two predicates: "Whole name" and "Whole address".
		       These predicates will produce the following blocks:

		       .. code :: python

		          # Block 1 (Whole name)
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (3, {'name' : 'Pat', 'address' : '123 Main'})

			  # Block 2 (Whole name)
			  (2, {'name' : 'Sam', 'address' : '123 Main'})

			  # Block 3 (Whole address
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Sam', 'address' : '123 Main'})
			  (3, {'name' : 'Pat', 'address' : '123 Main'})


		       So, the blocks you feed to matchBlocks should look
		       like this, 

		       .. code :: python

		          blocks =((
			            [(1, {'name' : 'Pat', 'address' : '123 Main'}, set([]))],
			            [(3, {'name' : 'Pat', 'address' : '123 Main'}, set([]))]
				    ), 
			           (
				    [(1, {'name' : 'Pat', 'address' : '123 Main'}, set([1])),
				     (2, {'name' : 'Sam', 'address' : '123 Main'}, set([]))],
			            [(3, {'name' : 'Pat', 'address' : '123 Main'}, set([1]))]
			            
				    )
				   )
			  linker.matchBlocks(blocks)

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
      

