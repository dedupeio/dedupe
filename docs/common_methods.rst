.. py:attribute:: learner

   By default, the learner is a stochastic gradient descent L2
   regularized logistic regression classifier. If you want to use a
   different classifier, you can overwrite this attribute with your
   custom function. This function must accept three arguments. 

   :param labels: A numpy array of 1 and 0's where a 1 indicates a
                  positive example and 0 a negative example. The array
		  should have dimensions of (num_examples, 1)
   :param examples: A numpy array of example vectors. The array should
		    have dimensions of (num_examples, num_features)
   :param float alpha: A regularizing constant for the classifier.

   The function must return a tuple of two elements

   :param weights: A list of weights, one for each feature.
   :param bias: A bias, or intercept term.

   .. code:: python

      from sklearn.linear_model import LogisticRegression

      def sklearner(labels, examples, alpha) :

	     learner = LogisticRegression(penalty='l2', C=1/alpha)
	     learner.fit(examples, labels)
	     weight, bias = list(learner.coef_[0]), learner.intercept_[0]
	     return weight, bias

      deduper = dedupe.Dedupe(fields)
      deduper.learner = sklearner
      

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

