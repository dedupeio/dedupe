=================
API Documentation
=================

:class:`Dedupe` Objects
--------------------------
Class for active learning deduplication. Use deduplication when you have
data that can contain multiple records that can all refer to the same
entity. 

.. py:class:: Dedupe(variable_definition, [data_sample=None, [num_cores]])

   Initialize a Dedupe object with a :doc:`field definition <Variable-definition>`

   :param dict variable_definition: A variable definition is list of 
				    dictionaries describing the variables
				    will be used for training a model.
   :param data_sample: is an optional argument that we discuss below
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine

   In order to learn how to deduplicate records, dedupe needs a sample
   of records you are trying to deduplicate. If your data is not too
   large (fits in memory), you can pass your data to the
   :py:meth:`~Dedupe.sample` method and dedupe will take a sample for
   you.

   .. code:: python

      # initialize from a defined set of fields
      variables = [
	           {'field' : 'Site name', 'type': 'String'},
		   {'field' : 'Address', 'type': 'String'},
		   {'field' : 'Zip', 'type': 'String', 'has missing':True},
		   {'field' : 'Phone', 'type': 'String', 'has missing':True}
		   ]

      deduper = dedupe.Dedupe(variables)

      deduper.sample(your_data)

   If your data won't fit in memory, you'll have to prepare a sample
   of the data yourself and pass it to Dedupe.

   ``data_sample`` should be a sequence of tuples, where each tuple
   contains a pair of records, and each record is a :py:class:`frozendict`
   object that contains the field names you declared in
   field\_definitions as keys.

   For example, a data_sample with only one pair of records,

   .. code:: python

      data_sample = [(
                      (dedupe.frozendict({'city': 'san francisco',
	                                  'address': '300 de haro st.',
		                          'name': "sally's cafe & bakery",
		                          'cuisine': 'american'}),
	               dedupe.frozendict({'city': 'san francisco',
	                                  'address': '1328 18th st.',
                                          'name': 'san francisco bbq',
                                          'cuisine': 'thai'})
	               )
	              ]

      deduper = dedupe.Dedupe(variables, data_sample)
      
   See `MySQL
   <http://open-city.github.com/dedupe/doc/mysql_example.html>`__ for
   an example of how to create a data sample yourself.

   .. py:method:: sample(data[, [sample_size=15000[, blocked_proportion=0.5]])

      If you did not initialize the Dedupe object with a data_sample, you
      will need to call this method to take a random sample of your data to be
      used for training.

      :param dict data: A dictionary-like object indexed by record ID
			where the values are dictionaries representing records.
      :param int sample_size: Number of record tuples to return. Defaults
			      to 15,000.
      :param float blocked_proportion: The proportion of record pairs to be sampled from similar records, as opposed to randomly selected pairs. Defaults to 0.5.

      .. code:: python

	 data_sample = deduper.sample(data_d, 150000, .5)



   .. include:: common_learning_methods.rst
   .. include:: common_dedupe_methods.rst
   .. include:: common_methods.rst


:class:`StaticDedupe` Objects
-----------------------------

Class for deduplication using saved settings. If you have already
trained dedupe, you can load the saved settings with StaticDedupe.

.. py:class:: StaticDedupe(settings_file, [num_cores])

   Initialize a Dedupe object with saved settings

   :param file settings_file: A file object containing settings info produced from
			      the :py:meth:`Dedupe.writeSettings` of a
			      previous, active Dedupe object.
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine


   .. code:: python
       with open('my_settings_file', 'rb') as f:
           deduper = StaticDedupe(f)

   .. include:: common_dedupe_methods.rst
   .. include:: common_methods.rst

:class:`RecordLink` Objects
---------------------------

Class for active learning record linkage.

Use RecordLinkMatching when you have two datasets that you want to
merge. Each dataset, individually, should contain no duplicates. A
record from the first dataset can match one and only one record from the
second dataset and vice versa. A record from the first dataset need not
match any record from the second dataset and vice versa.

For larger datasets, you will need to use the ``thresholdBlocks`` and
``matchBlocks``. This methods require you to create blocks of records.
For RecordLink, each blocks should be a pairs of dictionaries of
records. Each block consists of all the records that share a particular
predicate, as output by the blocker method of RecordLink.

Within a block, the first dictionary should consist of records from the
first dataset, with the keys being record ids and the values being the
record. The second dictionary should consist of records from the
dataset.

Example


.. code:: python

    > data_1 = {'A1' : {'name' : 'howard'}}
    > data_2 = {'B1' : {'name' : 'howie'}}
    ...
    > blocks = defaultdict(lambda : ({}, {}))
    >
    > for block_key, record_id in linker.blocker(data_1.items()) :
    >   blocks[block_key][0].update({record_id : data_1[record_id]})
    > for block_key, record_id in linker.blocker(data_2.items()) :
    >   if block_key in blocks :
    >     blocks[block_key][1].update({record_id : data_2[record_id]})
    >
    > blocked_data = blocks.values()
    > print blocked_data
    [({'A1' : {'name' : 'howard'}}, {'B1' : {'name' : 'howie'}})]


.. py:class:: RecordLink(variable_definition, [data_sample=None, [num_cores]])

   Initialize a Dedupe object with a variable definition

   :param dict variable_definition: A variable definition is list of 
				    dictionaries describing the variables
				    will be used for training a model.
   :param data_sample: is an optional argument that `we'll discuss fully
		       below <#wiki-sample-dedupe>`__
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine


   We assume that the fields you want to compare across datasets have the
   same field name.

   .. py:method:: sample(data_1, data_2, sample_size=150000, blocked_proportion=0.5)

      Draws a random sample of combinations of records from the first and
      second datasets, and initializes active learning with this sample

      :param dict data_1: A dictionary of records from first dataset,
			  where the keys are record_ids and the
			  values are dictionaries with the keys being
			  field names.
      :param dict data_2: A dictionary of records from second dataset,
			  same form as data_1
      :param int sample_size: The size of the sample to draw. Defaults to 150,000     
      :param float blocked_proportion: The proportion of record pairs to be sampled from similar records, as opposed to randomly selected pairs. Defaults to 0.5.

      .. code:: python

	  linker.sample(data_1, data_2, 150000)

   .. include:: common_recordlink_methods.rst
   .. include:: common_learning_methods.rst
   .. include:: common_methods.rst


:class:`StaticRecordLink` Objects
---------------------------------

Class for record linkage using saved settings. If you have already
trained a record linkage instance, you can load the saved settings with
StaticRecordLink.

.. py:class:: StaticRecordLink(settings_file, [num_cores])

   Initialize a Dedupe object with saved settings

   :param str settings_file: File object containing settings data produced from
			      the :py:meth:`RecordLink.writeSettings` of a
			      previous, active Dedupe object.
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine


   .. code:: python

       with open('my_settings_file', 'rb') as f:
           deduper = StaticDedupe(f)

   .. include:: common_recordlink_methods.rst
   .. include:: common_methods.rst

:class:`Gazetteer` Objects
---------------------------

Class for active learning gazetteer matching.

Gazetteer matching is for matching a messy data set against a
'canonical dataset', i.e. one that does not have any duplicates. This
class is useful for such tasks as matching messy addresses against
a clean list. 

The interface is the same as for RecordLink objects except for a
couple of methods.

.. py:class:: Gazetteer

   .. include:: common_gazetteer_methods.rst
   .. include:: common_learning_methods.rst
   .. include:: common_methods.rst



:class:`StaticGazetteer` Objects
--------------------------------

Class for gazetter matching using saved settings. If you have already
trained a gazetteer instance, you can load the saved settings with
StaticGazetteer.

This class has the same interface as StaticRecordLink except for a
couple of methods.

.. py:class:: StaticGazetteer

   .. include:: common_gazetteer_methods.rst
   .. include:: common_methods.rst



Convenience Functions
---------------------

.. py:function:: consoleLabel(matcher)

   Train a matcher instance (Dedupe or RecordLink) from the command line.
   Example

   .. code:: python

      > dedupe = Dedupe(variables, data_sample)
      > dedupe.consoleLabel(dedupe)

.. py:function:: trainingDataLink(data_1, data_2, common_key[, training_size])

   Construct training data for consumption by the
   :py:meth:`RecordLink.markPairs` from already linked datasets.

   :param dict data_1: a dictionary of records from first dataset,
		       where the keys are record_ids and the
		       values are dictionaries with the keys being
		       field names.
   :param dict data_2: a dictionary of records from second dataset,
		       same form as data_1
   :param str common_key: the name of the record field that uniquely
			 identifies a match
   :param int training_size: the rough limit of the number of training examples,
			     defaults to 50000

   **Warning**

   Every match must be identified by the sharing of a common key. This
   function assumes that if two records do not share a common key then they
   are distinct records.

.. py:function:: trainingDataDedupe(data, common_key[, training_size])

   Construct training data for consumption by the
   :py:meth:`Dedupe.markPairs` from an already deduplicated dataset.

   :param dict data: a dictionary of records, where the keys are
		     record_ids and the values are dictionaries with
		     the keys being field names
   :param str common_key: the name of the record field that uniquely
			 identifies a match
   :param int training_size: the rough limit of the number of training examples,
			     defaults to 50000


   **Warning**

   Every match must be identified by the sharing of a common key. This
   function assumes that if two records do not share a common key then
   they are distinct records.


.. py:function:: canonicalize(record_cluster)
   
   Constructs a canonical representation of a duplicate cluster by finding canonical values for each field

   :param list record_cluster: A list of records within a duplicate cluster, where the records are dictionaries with field 
                  names as keys and field values as values

   .. code:: python

.. py:function:: randomPairs(n_records, sample_size)

   If you have N records there are :math:`\frac{N(N-1)}{2}` unique
   pairs of records (where each record is different and order doesn't
   matter). If we indexed the N records from 0 to N-1, we would have
   :math:`\frac{N(N-1)}{2}` corresponding pairs of indices ::
   
      (0, 1)
      (0, 2)
      ...
      (0, N-2)
      (0, N-1)
      (1, 2)
      (1, 3)
      ...
      (N-3, N-2)
      (N-3, N-1)
      (N-2, N-1)

   randomPairs returns a random sample from the set of unique pairs of
   indices. The function attempts to draw the sample without
   replacement, but may draw a sample with replacement. If that
   happens, you will be warned.

   This can be useful when you need to create a sample of pairs from
   your data, but you don't want to pass all of your data into
   :py:meth:`~Dedupe.sample` because, for instance, all your data is
   too big to fit into memory.

   :param int n_record: the number of records in your record set

   :param int sample_size: the size of sample you desire
      
.. py:function:: randomPairsMatch(n_records_a, n_records_b, sample_size)

   If you have two record sets of length N and M, there are :math:`NM`
   unique pairs of records (where each record is from a different
   record set and order doesn't matter). If we indexed the N records
   from 0 to N-1, we would have :math:`NM` corresponding pairs of
   indices ::

       (0, 0)
       (0, 1)
       ...
       (0, M-1)
       (1, 0)
       (1, 1)
       ...
       (N-1, 0)
       (N-1, 1)
       ...
       (N-1, M-1)
 
   randomPairs returns a random sample from the set of unique pairs of
   indices. The function attempts to draw the sample without
   replacement, but may draw a sample with replacement. If that
   happens, you will be warned.

   This can be useful when you need to create a sample of pairs from
   your data, but you don't want to pass all of your data into
   :py:meth:`~Dedupe.sample` because, for instance, all your data is
   too big to fit into memory.

   :param int n_record_a: the number of records in your first record set

   :param int n_record_b: the number of records in your second record set

   :param int sample_size: the size of sample you desire

.. py:class:: frozendict(d)
  
   Initialize a frozendict object. `frozendicts` are like normal
   python dictionaries except 1. you can't change them and 2. you can
   hash them. We depend on the hashing in a few places when we are
   training Dedupe. 

   :param dict d: a dictionary, typically a dictionary representing
                  your record
