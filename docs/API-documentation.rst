=================
Library Documentation
=================

:class:`Dedupe` Objects
--------------------------
Class for active learning deduplication. Use deduplication when you have
data that can contain multiple records that can all refer to the same
entity. 

.. py:class:: Dedupe(variable_definition, [data_sample[, [num_cores]])

   Initialize a Dedupe object with a :doc:`field definition <Variable-definition>`

   :param dict variable_definition: A variable definition is list of 
				    dictionaries describing the variables
				    will be used for training a model.
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine

   :param data_sample: __DEPRECATED__

   .. code:: python

      # initialize from a defined set of fields
      variables = [
	           {'field' : 'Site name', 'type': 'String'},
		   {'field' : 'Address', 'type': 'String'},
		   {'field' : 'Zip', 'type': 'String', 'has missing':True},
		   {'field' : 'Phone', 'type': 'String', 'has missing':True}
		   ]

      deduper = dedupe.Dedupe(variables)

   .. py:method:: sample(data[, [sample_size=15000[, blocked_proportion=0.5[, original_length]]])
		  
   In order to learn how to deduplicate your records, dedupe needs a
   sample of your records to train on. This method takes a mixture of
   random sample of pairs of records and a selection of pairs of
   records that are much more likely to be duplicates.
		  
   :param dict data: A dictionary-like object indexed by record ID
		     where the values are dictionaries representing records.
   :param int sample_size: Number of record tuples to return. Defaults
			   to 15,000.
   :param float blocked_proportion: The proportion of record pairs
                                    to be sampled from similar
                                    records, as opposed to randomly
                                    selected pairs. Defaults to
                                    0.5.
   :param original_length: If `data` is a subsample of all your data,
                           `original_length` should be the size of
                           your complete data. By default,
                           `original_length` defaults to the length of
                           `data`.
				       
   .. code:: python

      deduper.sample(data_d, 150000, .5)


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


.. py:class:: RecordLink(variable_definition, [data_sample, [[num_cores]])

   Initialize a Dedupe object with a variable definition

   :param dict variable_definition: A variable definition is list of 
				    dictionaries describing the variables
				    will be used for training a model.
   :param int num_cores: the number of cpus to use for parallel
			 processing, defaults to the number of cpus
			 available on the machine
   :param data_sample: __DEPRECATED__

   We assume that the fields you want to compare across datasets have the
   same field name.

   .. py:method:: sample(data_1, data_2, [sample_size=150000[, blocked_proportion=0.5, [original_length_1[, original_length_2]]]])

   In order to learn how to link your records, dedupe needs a
   sample of your records to train on. This method takes a mixture of
   random sample of pairs of records and a selection of pairs of
   records that are much more likely to be duplicates.

   :param dict data_1: A dictionary of records from first dataset,
		       where the keys are record_ids and the
		       values are dictionaries with the keys being
		       field names.
   :param dict data_2: A dictionary of records from second dataset,
		       same form as data_1
   :param int sample_size: The size of the sample to draw. Defaults to 150,000     
   :param float blocked_proportion: The proportion of record pairs to
                                    be sampled from similar records,
                                    as opposed to randomly selected
                                    pairs. Defaults to 0.5.
   :param original_length_1: If `data_1` is a subsample of your first dataset,
                             `original_length_1` should be the size of
                             the complete first dataset. By default,
                             `original_length_1` defaults to the length of
                             `data_1`
   :param original_length_2: If `data_2` is a subsample of your first dataset,
                             `original_length_2` should be the size of
                             the complete first dataset. By default,
                             `original_length_2` defaults to the length of
                             `data_2`
				    
   

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

      > deduper = dedupe.Dedupe(variables)
      > deduper.sample(data)
      > dedupe.consoleLabel(deduper)

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

