.. py:attribute:: classifier

   By default, the classifier is a `L2 regularized logistic regression
   classifier <https://pypi.python.org/pypi/rlr>`. If you want to use
   a different classifier, you can overwrite this attribute with your
   custom object. Your classifier object must be have `fit` and
   `predict_proba` methods, like `sklearn models
   <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`.

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
		  
