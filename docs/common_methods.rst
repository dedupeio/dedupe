.. py:method:: thresholdBlocks(blocks, recall_weight=1.5)

   Returns the threshold that maximizes the expected F score, a weighted
   average of precision and recall for a sample of blocked data.

   For larger datasets, you will need to use the ``thresholdBlocks``
   and ``matchBlocks``. This methods require you to create blocks of
   records.  For Dedupe, each blocks should be a dictionary of
   records. Each block consists of all the records that share a
   particular predicate, as output by the blocker method of Dedupe.

   Within a block, the dictionary should consist of records from the data,
   with the keys being record ids and the values being the record.

   .. code:: python

      > data = {'A1' : {'name' : 'howard'},
		'B1' : {'name' : 'howie'}}
      ...
      > blocks = defaultdict(dict)
      >
      > for block_key, record_id in linker.blocker(data_d.items()) :
      >   blocks[block_key].update({record_id : data_d[record_id]})
      >
      > blocked_data = blocks.values()
      > print blocked_data
      [{'A1' : {'name' : 'howard'},
	'B1' : {'name' : 'howie'}}]


   .. code:: python

       threshold = deduper.thresholdBlocks(blocked_data, recall_weight=2)

   Keyword arguments

   ``blocks`` Sequence of tuples of records, where each tuple is a set of
   records covered by a blocking predicate.

   ``recall_weight`` Sets the tradeoff between precision and recall. I.e.
   if you care twice as much about recall as you do precision, set
   recall\_weight to 2.

.. py:method::  matchBlocks(blocks, threshold=.5)

   Partitions blocked data and returns a list of clusters, where each
   cluster is a tuple of record ids

   .. code:: python

       clustered_dupes = deduper.matchBlocks(blocked_data, threshold)

   Keyword arguments

   ``blocks`` Sequence of tuples of records, where each tuple is a set of
   records covered by a blocking predicate.

   ``threshold`` Number between 0 and 1 (default is .5). We will only
   consider as duplicates record pairs as duplicates if their estimated
   duplicate likelihood is greater than the threshold.

   Lowering the number will increase recall, raising it will increase
   precision.
