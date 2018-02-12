.. py:method:: index(data) 

   Add records to the index of records to match against. If a record in
   `canonical_data` has the same key as a previously indexed record, the 
   old record will be replaced.

   :param dict data: a dictionary of records where the keys
		     are record_ids and the values are
		     dictionaries with the keys being
		     field_names

.. py:method:: unindex(data) :
   
   Remove records from the index of records to match against. 

   :param dict data: a dictionary of records where the keys
		     are record_ids and the values are
		     dictionaries with the keys being
		     field_names


.. py:method:: match(messy_data, [threshold=0.5[, n_matches=1[, generator=False]]])

   Identifies pairs of records that could refer to the same entity,
   returns tuples containing tuples of possible matches, with a
   confidence score for each match. The record_ids within each tuple
   should refer to potential matches from a messy data record to
   canonical records. The confidence score is the estimated
   probability that the records refer to the same entity.

   :param dict messy_data: a dictionary of records from a messy
			   dataset, where the keys are record_ids and
			   the values are dictionaries with the keys
			   being field names.

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
			 messy_data. If set to `None` all possible
			 matches above the threshold will be
			 returned. Defaults to 1

   :param bool generator: when `True`, match will generate a sequence of
			  possible matches, instead of a list. Defaults
			  to `False` This makes `match` a lazy method.

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

.. py:method::  threshold(messy_data, recall_weight = 1.5) 

   Returns the threshold that maximizes the expected F score, a
   weighted average of precision and recall for a sample of data.

   :param dict messy_data: a dictionary of records from a messy
			   dataset, where the keys are record_ids and
			   the values are dictionaries with the keys
			   being field names.

   :param float recall_weight: Sets the tradeoff between precision and
                               recall. I.e. if you care twice as much
                               about recall as you do precision, set
                               recall_weight to 2.


.. py:method::  matchBlocks(blocks, threshold=.5, n_matches=1)

   Partitions blocked data and returns a list of clusters, where each
   cluster is a tuple of record ids

   :param list blocks: Sequence of records blocks. Each record block
		       is a tuple containing two sequences of records,
		       the records from the messy data set and the
		       records from the canonical dataset. Within each
		       block there should be at least one record from
		       each datasets.  Along with each record, there
		       should also be information on the blocks that
		       cover that record.

		       For example, if we have two records from a 
		       messy dataset one record from a canonical dataset: 

		       .. code :: python
		           
		          # Messy
		          (1, {'name' : 'Pat', 'address' : '123 Main'})
			  (2, {'name' : 'Sam', 'address' : '123 Main'})

			  # Canonical
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
			            [(3, {'name' : 'Pat', 'address' : '123 Main'}, set([])]
				    ), 
			           (
				    [(1, {'name' : 'Pat', 'address' : '123 Main'}, set([1]),
				     ((2, {'name' : 'Sam', 'address' : '123 Main'}, set([])],
			            [((3, {'name' : 'Pat', 'address' : '123 Main'}, set([1])]
			            
				    )
				   )
			  linker.matchBlocks(blocks)

   :param float threshold: Number between 0 and 1 (default is .5). We
			   will only consider as duplicates record
			   pairs as duplicates if their estimated
			   duplicate likelihood is greater than the
			   threshold.

			   Lowering the number will increase recall,
			   raising it will increase precision.

   :param int n_matches: the maximum number of possible matches from
			 canonical_data to return for each record in
			 messy_data. If set to `None` all possible
			 matches above the threshold will be
			 returned. Defaults to 1


   .. code:: python

       clustered_dupes = deduper.matchBlocks(blocked_data, threshold)

