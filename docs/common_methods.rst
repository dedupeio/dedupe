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

