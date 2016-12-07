.. py:method:: uncertainPairs()

   Returns a list of pairs of records from the sample of record pairs
   tuples that Dedupe is most curious to have labeled.

   This method is mainly useful for building a user interface for training
   a matching model.

    .. code:: python

       > pair = deduper.uncertainPairs()
       > print pair
       [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]


.. py:method:: markPairs(labeled_examples)

   Add users labeled pairs of records to training data and update the
   matching model

   This method is useful for building a user interface for training a
   matching model or for adding training data from an existing source.

   :param dict labeled_examples: a dictionary with two keys,
				  ``match`` and ``distinct`` the
				  values are lists that can contain
				  pairs of records.

   .. code:: python

      labeled_examples = {'match'    : [], 
			 'distinct' : [({'name' : 'Georgie Porgie'}, 
					{'name' : 'Georgette Porgette'})]
			 }
      deduper.markPairs(labeled_examples)


.. py:method:: train([recall=0.95, [index_predicates=True]])

   Learn final pairwise classifier and blocking rules. Requires that
   adequate training data has been already been provided.


   :param float recall: The proportion of true dupe pairs in our
			training data that that we the learned blocks
			must cover. If we lower the recall, there will
			be pairs of true dupes that we will never
			directly compare.

			recall should be a float between 0.0 and 1.0,
			the default is 0.95

   :param bool index_predicates: Should dedupe consider predicates
				 that rely upon indexing the
				 data. Index predicates can be slower
				 and take susbstantial memory.

				 Defaults to True.

   .. code:: python

      deduper.train()


.. py:method:: writeTraining(file_obj)

   Write json data that contains labeled examples to a file object.

   :param file file_obj: File object.

   .. code:: python

      with open('./my_training.json', 'w') as f:
          deduper.writeTraining(f)

.. py:method:: readTraining(training_file)

   Read training from previously saved training data file object

   :param file training_file: File object containing training data

   .. code:: python

      with open('./my_training.json') as f:
          deduper.readTraining(f)

.. py:method:: cleanupTraining()

   Delete data we used for training.

   ``data_sample``, ``training_pairs``, ``training_data``, and
   ``activeLearner`` can be very large objects. When you are done
   training you may want to free up the memory they use.
   
   .. code:: python

      deduper.cleanupTraining()
