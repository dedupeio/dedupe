.. py:attribute:: classifier

   By default, the classifier is a `L2 regularized logistic regression
   classifier <https://pypi.python.org/pypi/rlr>`_. If you want to use
   a different classifier, you can overwrite this attribute with your
   custom object. Your classifier object must be have `fit` and
   `predict_proba` methods, like `sklearn models
   <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

   .. code:: python

      from sklearn.linear_model import LogisticRegression

      deduper = dedupe.Dedupe(fields)
      deduper.classifier = LogisticRegression()
      

.. py:method:: thresholdBlocks(blocks, recall_weight=1.5)

   Returns the threshold that maximizes the expected F score, a weighted
   average of precision and recall for a sample of blocked data.

   For larger datasets, you will need to use the ``thresholdBlocks``
   and ``matchBlocks``. This methods require you to create blocks of
   records.  See the documentation for the ``matchBlocks`` method
   for how to construct blocks. 
   .. code:: python

       threshold = deduper.thresholdBlocks(blocked_data, recall_weight=2)

   Keyword arguments

   :param list blocks: See ```matchBlocks```

   :param float recall_weight: Sets the tradeoff between precision and
			       recall. I.e.  if you care twice as much
			       about recall as you do precision, set
			       recall\_weight to 2.

.. py:method::  matchBlocks(blocks, [threshold=.5])

   Partitions blocked data and generates a sequence of clusters, where each
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


.. py:method:: writeSettings(file_obj, [index=False])

   Write a settings file that contains the data model and predicates
   to a file object.

   :param file file_obj: File object.
   :param index bool: Should the indexes of index predicates be
                        saved. You will probably only want to call
                        this after indexing all of your records.
			 

   .. code:: python

      with open('my_learned_settings', 'wb') as f:
          deduper.writeSettings(f, indexes=True)

			       
.. py:attribute:: loaded_indices

   Indicates whether indices for index predicates was loaded from a
   settings file.
		  
.. py:method:: blocker(data[, target=False])

   Generate the predicates for records. Yields tuples of (predicate,
   record_id).

   :param list data: A sequence of tuples of (record_id,
                     record_dict). Can often be created by
                     `data_dict.items()`.
   :param bool target: Indicates whether the data should be treated as
		       the target data. This effects the behavior of
		       search predicates. If `target` is set to
		       `True`, an search predicate will return the
		       value itself. If `target` is set to `False` the
		       search predicate will return all possible
		       values within the specified search distance.

		       Let's say we have a
		       `LevenshteinSearchPredicate` with an associated
		       distance of `1` on a `"name"` field; and we
		       have a record like `{"name": "thomas"}`. If the
		       `target` is set to `True` then the predicate
		       will return `"thomas"`.  If `target` is set to
		       `False`, then the blocker could return
		       `"thomas"`, `"tomas"`, and `"thoms"`. By using
		       the `target` argument on one of your datasets,
		       you will dramatically reduce the total number
		       of comparisons without a loss of accuracy.
		       

   .. code:: python

      > data = [(1, {'name' : 'bob'}), (2, {'name' : 'suzanne'})]
      > blocked_ids = deduper.blocker(data)
      > print list(blocked_ids)
      [('foo:1', 1), ..., ('bar:1', 100)]


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
	     
