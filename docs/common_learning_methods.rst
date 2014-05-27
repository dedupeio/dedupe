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

      labeled_example = {'match'    : [], 
			 'distinct' : [({'name' : 'Georgie Porgie'}, 
					{'name' : 'Georgette Porgette'})]
			 }
      deduper.markPairs(labeled_examples)


.. py:method:: train([ppc=1.0, [uncovered_dupes=1]])

   Learn final pairwise classifier and blocking rules. Requires that
   adequate training data has been already been provided.

   :param float ppc: Limits the Proportion of Pairs Covered that we
		      allow a predicate to cover. If a predicate puts
		      together a fraction of possible pairs greater
		      than the ppc, that predicate will be removed
		      from consideration.

		      As the size of the data increases, the user will
		      generally want to reduce ppc.

		      ppc should be a value between 0.0 and 1.0

   :param int uncovered_dupes: The number of true dupes pairs in our
				training data that we can accept will
				not be put into any block. If true
				true duplicates are never in the same
				block, we will never compare them, and
				may never declare them to be
				duplicates.

				However, requiring that we cover every
				single true dupe pair may mean that we
				have to use blocks that put together
				many, many distinct pairs that we'll
				have to expensively, compare as well.

   .. code:: python

      deduper.train()


.. py:method:: writeTraining(file_name)

   Write to a json file that contains labeled examples.

   :param str file_name: Path to a json file.

   .. code:: python

      deduper.writeTraining('./my_training.json')

.. py:method:: readTraining(training_source)

   Read training from previously saved training data file

   :param str training_source: the path of a training data file

   .. code:: python

      deduper.readTraining('./my_training.json')

.. py:method:: writeSettings(file_name)

   Write a settings file that contains the data model and predicates

   :param str file_name: Path to file.

   .. code:: python

      deduper.writeSettings('my_learned_settings')
