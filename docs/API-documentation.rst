=================
API Documentation
=================

`Dedupe <#dedupe-1>`__
~~~~~~~~~~~~~~~~~~~~~~

-  `**Defining a model** <#wiki-init-dedupe-active>`__
-  `**sample** <#wiki-sample-dedupe>`__
-  `uncertainPairs <#wiki-uncertainPairs>`__
-  `markPairs <#wiki-markPairs>`__
-  `**train** <#wiki-train>`__
-  `**threshold** <#wiki-threshold-dedupe>`__
-  `**match** <#wiki-match-dedupe>`__
-  `**writeTraining** <#wiki-writeTraining>`__
-  `**readTraining** <#wiki-readTraining>`__
-  `**writeSettings** <#wiki-writeSettings>`__
-  `blocker <#wiki-blocker>`__
-  `thresholdBlocks <#wiki-thresholdBlocks>`__
-  `matchBlocks <#wiki-matchBlocks>`__

`StaticDedupe <#staticdedupe-1>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `**Loading settings** <#wiki-init-dedupe-static>`__
-  `threshold <#wiki-threshold-dedupe>`__
-  `match <#wiki-match-dedupe>`__
-  `blocker <#wiki-blocker>`__
-  `thresholdBlocks <#wiki-thresholdBlocks>`__
-  `matchBlocks <#wiki-matchBlocks>`__

`RecordLink <#recordlink-1>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Defining a model <#wiki-init-recordlink-active>`__
-  `sample <#wiki-sample-recordlink>`__
-  `uncertainPairs <#wiki-uncertainPairs>`__
-  `markPairs <#wiki-markPairs>`__
-  `train <#wiki-train>`__
-  `threshold <#wiki-threshold-recordlink>`__
-  `match <#wiki-match-recordlink>`__
-  `readTraining <#wiki-readTraining>`__
-  `writeSettings <#wiki-writeSettings>`__
-  `writeTraining <#wiki-writeTraining>`__
-  `blocker <#wiki-blocker>`__
-  `thresholdBlocks <#wiki-thresholdBlocks>`__
-  `matchBlocks <#wiki-matchBlocks>`__

`StaticRecordLink <#staticrecordlink-1>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Loading settings <#wiki-init-dedupe-static>`__
-  `threshold <#wiki-threshold-dedupe-recordlink>`__
-  `match <#wiki-match-dedupe-recordlink>`__
-  `blocker <#wiki-blocker>`__
-  `thresholdBlocks <#wiki-thresholdBlocks>`__
-  `matchBlocks <#wiki-matchBlocks>`__

`Convenience <#convenience-1>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `consoleLabel <#wiki-consoleLabel>`__
-  `trainingDataLink <#wiki-trainingDataLink>`__
-  `trainingDataDedupe <#wiki-trainingDedupe>`__

Dedupe
======

Class for active learning deduplication. Use deduplication when you have
data that can contain multiple records that can all refer to the same
entity. Active learning lets a user train dedupe on how to classify
records as distinct or duplicates and efficiently learn how to match
records.

For larger datasets, you will need to use the ``thresholdBlocks`` and
``matchBlocks``. This methods require you to create blocks of records.
For Dedupe, each blocks should be a dictionary of records. Each block
consists of all the records that share a particular predicate, as output
by the blocker method of Dedupe.

Within a block, the dictionary should consist of records from the data,
with the keys being record ids and the values being the record.

Example
-------

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

Defining a model, \_\ *init\_*\ (field\_definition, data\_sample=None)
----------------------------------------------------------------------

Initialize a Dedupe object with a field definition

.. code:: python

    # initialize from a defined set of fields
    fields = {
              'Site name': {'type': 'String'},
              'Address': {'type': 'String'},
              'Zip': {'type': 'String', 'Has Missing':True},
              'Phone': {'type': 'String', 'Has Missing':True},
              }

    deduper = dedupe.Dedupe(fields)

or ``deduper = dedupe.Dedupe(fields, data_sample)``

``data_sample`` is an optional argument that `we'll discuss fully
below <#wiki-sample-dedupe>`__

``num_processes`` should be the number of processes to use for parallel
processing, defaults to 1

Field Definitions
~~~~~~~~~~~~~~~~~

A field definition is a dictionary where the keys are the fields that
will be used for training a model and the values are the field
specification

Field types include \* String \* Custom \* LatLong \* Set \* Interaction

String Types
^^^^^^^^^^^^

A 'String' type field must have as its key a name of a field as it
appears in the data dictionary and a type declaration ex.
``{'Phone': {type: 'String'}}`` The string type expects fields to be of
class string. Missing data should be represented as an empty string
``''``

String types are compared using `affine gap string
distance <http://en.wikipedia.org/wiki/Gap_penalty#Affine>`__.

Custom Types
^^^^^^^^^^^^

A 'Custom' type field must have as its key a name of a field as it
appears in the data dictionary, at 'type' declaration, and a
'comparator' declaration. The comparator must be a function that can
take in two field values and return a number or a numpy.nan (not a
number, appropriate when a distance is not well defined, as when one of
the fields is missing).

Example custom comparator:
``python def sameOrNotComparator(field_1, field_2) :     if field_1 and field_2 :         if field_1 == field_2 :             return 1         else:             return 0     else :         return numpy.nan``

Field definition:

.. code:: python

    {'Zip': {'type': 'Custom', 
             'comparator' : sameOrNotComparator}} 

LatLong
^^^^^^^

A 'LatLong' type field must have as its key a name of a field as it
appears in the data dictionary, at 'type' declaration. LatLong fields
are compared using the `Haversine
Formula <http://en.wikipedia.org/wiki/Haversine_formula>`__. A 'LatLong'
type field must consist of tuples of floats corresponding to a latitude
and a longitude. If data is missing, this should be represented by a
tuple of 0s ``(0.0, 0.0)``

.. code:: python

    {'Location': {'type': 'LatLong'}} 

Set
^^^

A 'Set' type field must have as its key a name of a field as it appears
in the data dictionary, at 'type' declaration. Set fields are compares
sets using the `Jaccard
index <http://en.wikipedia.org/wiki/Jaccard_index>`__. Missing data is
on implemented for this field type.

.. code:: python

    {'Co-authors': {'type': 'Set'}} 

Interaction
^^^^^^^^^^^

An interaction type field can have as it's key any name you choose, a
'type' declaration, and an 'Interaction Fields' declaration. An
interaction field multiplies the values of the declared fields.

The 'Interaction Fields' must be a sequence of names of other fields you
have defined in your field definition.

`Interactions <http://en.wikipedia.org/wiki/Interaction_%28statistics%29>`__
are good when the effect of two predictors is not simply additive.

.. code:: python

    {'Name'     : {'type': 'String'}, 
     'Zip'      : {'type': 'Custom', 
                   'comparator' : sameOrNotComparator},
     'Name-Zip' : {'type': 'Interaction', 
                   'Interaction Fields' : ['Name', 'Zip']}} 

Categorical
^^^^^^^^^^^

Categorical variables are useful when you are dealing with qualitatively
different types of things. For example, you may have data on businesses
and you find that taxi cab businesses tend to have very similar names
but law firms don't. Categorical variables would let you indicate
whether two records are both taxi companies, both law firms, or one of
each.

Dedupe would represents these three possibilities using two dummy
variables:

::

    taxi-taxi      0 0
    lawyer-lawyer  1 0
    taxi-lawyer    0 1

A categorical field declaration must include a list of all the different
strings that you want to treat as different categories.

So if you data looks like this
``'Name'          'Business Type' AAA Taxi        taxi AA1 Taxi        taxi Hindelbert Esq  lawyer``
You would create a definition like:

.. code:: python

    {'Business Type'    : {'type': 'Categorical',
                           'Categories' : ['taxi', 'lawyer']}}

Source
^^^^^^

Usually different data sources vary in how many duplicates are contained
within them and the patterns that make two pairs of records likely to be
duplicates. If you are trying to link records from more than one data
set, it can be useful to take these differences into account.

If your data has a field that indicates its source, something like

::

    'Name'         'Source'
    John Adams     Campaign Contributions
    John Q. Adams  Lobbyist Registration
    John F. Adams  Lobbyist Registration

You can take these sources into account by the following field
definition.

.. code:: python

    {'Source'    : {'type': 'Source',
                    'Categories' : ['Campaign Contributions', 'Lobbyist Registration']}}

Dedupe will create a categorical variable for the source and then
cross-interact it with all the other variables. This has the effect of
letting dedupe learn three different models at once. Let's say that we
had defined another variable called name. Then our total model would
have the following fields

::

    bias
    Name
    Source
    Source:Name
    different sources
    different sources:Name

``Bias + Name`` would predict the probability that a pair of records
were duplicates if both records were from ``Campaign Contributions``.

``Bias + Source + Name + Source:Name`` would predict the probability
that a pair of records were duplicates if both records were from
``Lobbyist Registration``

``Bias + different sources + Name + different sources:Name`` would
predict the probability that a pair of records were duplicates if one
record was from each of the two sources.

Missing Data
~~~~~~~~~~~~

If a field has missing data, you can set ``'Has Missing' : True`` in the
field definition. This creates a new, additional field representing
whether the data was present or not and zeros out the missing data. If
there is missing data, but you did not declare ``'Has Missing' : True``
then the missing data will simply be zeroed out.

If you define an an interaction with a field that you declared to have
missing data, then ``Has Missing : True`` will also be set for the
Interaction field.

Longer example of a field definition:

.. code:: python

    fields = {'name'         : {'type' : 'String'},
              'address'      : {'type' : 'String'},
              'city'         : {'type' : 'String'},
              'zip'          : {'type' : 'Custom', 'comparator' : sameOrNotComparator},
              'cuisine'      : {'type' : 'String', 'Has Missing': True}
              'name-address' : {'type' : 'Interaction', 'Interaction Fields' : ['name', 'city']}
              }

sample(data, sample\_size=150000)
---------------------------------

If you did not initialize the Dedupe object with a data\_sample, you
will need to call this method to take a random sample of your data to be
used for training.

Example usage
~~~~~~~~~~~~~

.. code:: python

    data_sample = deduper.sample(data_d, 150000)

Keyword arguments
~~~~~~~~~~~~~~~~~

``data`` A dictionary-like object indexed by record ID where the values
are dictionaries representing records.

``sample_size`` Number of record tuples to return. Defaults to 150,000.

If can't use this method because of the size your data (see the
`MySQL <http://open-city.github.com/dedupe/doc/mysql_example.html>`__),
you'll need to initialize Dedupe with the sample

.. code:: python

    deduper = Dedupe.(field_definition, data_sample)

``data_sample`` should be a sequence of tuples, where each tuple
contains a pair of records, and each record is a dictionary-like object
that contains the field names you declared in field\_definitions as
keys.

For example, a data\_sample with only one pair of records,

.. code:: python

    [
      (
       (854, {'city': 'san francisco',
              'address': '300 de haro st.',
              'name': "sally's cafe & bakery",
              'cuisine': 'american'}),
       (855, {'city': 'san francisco',
             'address': '1328 18th st.',
             'name': 'san francisco bbq',
             'cuisine': 'thai'})
       )
     ]

uncertainPairs()
----------------

Returns a list of pairs of records from the sample of record pairs
tuples that Dedupe is most curious to have labeled.

Example usage
~~~~~~~~~~~~~

.. code:: python

    > pair = deduper.uncertainPairs()
    > print pair
    [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]

This method is mainly useful for building a user interface for training
a matching model.

markPairs(labeled\_examples)
----------------------------

Add users labeled pairs of records to training data and update the
matching model

``labeled_examples`` must be a dictionary with two keys, ``match`` and
``distinct`` the values are lists that can contain pairs of records.

Example usage
~~~~~~~~~~~~~

.. code:: python

    labeled_example = {'match'    : [], 
                       'distinct' : [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]
                       }
    deduper.markPairs(labeled_examples)

This method is useful for building a user interface for training a
matching model or for adding training data from an existing source.

train(ppc=1, uncovered\_dupes=1)
--------------------------------

Learn final pairwise classifier and blocking rules. Requires that
adequate training data has been already been provided.

Keyword arguments
~~~~~~~~~~~~~~~~~

``ppc`` Limits the Proportion of Pairs Covered that we allow a predicate
to cover. If a predicate puts together a fraction of possible pairs
greater than the ppc, that predicate will be removed from consideration.

As the size of the data increases, the user will generally want to
reduce ppc.

ppc should be a value between 0.0 and 1.0

``uncovered_dupes`` The number of true dupes pairs in our training data
that we can accept will not be put into any block. If true true
duplicates are never in the same block, we will never compare them, and
may never declare them to be duplicates.

However, requiring that we cover every single true dupe pair may mean
that we have to use blocks that put together many, many distinct pairs
that we'll have to expensively, compare as well.

Example usage
~~~~~~~~~~~~~

.. code:: python

    deduper.train()

threshold(data, recall\_weight=1.5)
-----------------------------------

Returns the threshold that maximizes the expected F score, a weighted
average of precision and recall for a sample of data.

Arguments
~~~~~~~~~

``data`` is a dictionary of records, where the keys are record\_ids and
the values are dictionaries with the keys being field names

``recall_weight`` sets the tradeoff between precision and recall. I.e.
if you care twice as much about recall as you do precision, set
recall\_weight to 2.

Example usage
~~~~~~~~~~~~~

.. code:: python

    > threshold = deduper.threshold(data, recall_weight=2)
    > print threshold
    0.21

match(data, threshold = 0.5)
----------------------------

Identifies records that all refer to the same entity, returns tuples of
record ids, where the record\_ids within each tuple should refer to the
same entity

This method should only used for small to moderately sized datasets for
larger data, use matchBlocks

Arguments:
~~~~~~~~~~

``data`` is a dictionary of records, where the keys are record\_ids and
the values are dictionaries with the keys being field names

``threshold`` is a number between 0 and 1 (default is .5). We will
consider records as potential duplicates if the predicted probability of
being a duplicate is above the threshold.

Lowering the number will increase recall, raising it will increase
precision

Example usage
~~~~~~~~~~~~~

.. code:: python

    > duplicates = deduper.match(data, threshold=0.5)
    > print duplicates
    [(3,6,7), (2,10), ..., (11,14)]

writeTraining(file\_name)
-------------------------

Write to a json file that contains labeled examples.

Keyword arguments
~~~~~~~~~~~~~~~~~

``file_name`` Path to a json file.

Example usage
~~~~~~~~~~~~~

.. code:: python

    deduper.writeTraining('./my_training.json')

readTraining(training\_source)
------------------------------

Read training from previously saved training data file

Arguments:
~~~~~~~~~~

``training_source`` is the path of the training data file

Example usage
~~~~~~~~~~~~~

.. code:: python

    deduper.readTraining('./my_training.json')

writeSettings(file\_name)
-------------------------

Write a settings file that contains the data model and predicates

Keyword arguments
~~~~~~~~~~~~~~~~~

``file_name`` Path to file.

Example usage
~~~~~~~~~~~~~

.. code:: python

    deduper.writeSettings('my_learned_settings')

blocker(data)
-------------

Generate the predicates for records. Yields tuples of (predicate,
record\_id).

Arguments
~~~~~~~~~

``data`` A dictionary-like object indexed by record ID where the values
are dictionaries representing records.

Example usage
~~~~~~~~~~~~~

.. code:: python

    > blocked_ids = deduper.blocker(data)
    > print list(blocked_ids)
    [('foo:1', 1), ..., ('bar:1', 100)]

thresholdBlocks(blocks, recall\_weight=1.5)
-------------------------------------------

Returns the threshold that maximizes the expected F score, a weighted
average of precision and recall for a sample of blocked data.

Example usage
~~~~~~~~~~~~~

.. code:: python

    threshold = deduper.thresholdBlocks(blocked_data, recall_weight=2)

Keyword arguments
~~~~~~~~~~~~~~~~~

``blocks`` Sequence of tuples of records, where each tuple is a set of
records covered by a blocking predicate.

``recall_weight`` Sets the tradeoff between precision and recall. I.e.
if you care twice as much about recall as you do precision, set
recall\_weight to 2.

matchBlocks(blocks, threshold=.5)
---------------------------------

Partitions blocked data and returns a list of clusters, where each
cluster is a tuple of record ids

Example usage
~~~~~~~~~~~~~

.. code:: python

    clustered_dupes = deduper.matchBlocks(blocked_data, threshold)

Keyword arguments
~~~~~~~~~~~~~~~~~

``blocks`` Sequence of tuples of records, where each tuple is a set of
records covered by a blocking predicate.

``threshold`` Number between 0 and 1 (default is .5). We will only
consider as duplicates record pairs as duplicates if their estimated
duplicate likelihood is greater than the threshold.

Lowering the number will increase recall, raising it will increase
precision.

StaticDedupe
============

Class for deduplication using saved settings. Use deduplication when you
have data that can contain multiple records that can all refer to the
same entity. If you have already trained dedupe, you can load the saved
settings with StaticDedupe.

Loading Settings, \_\ *init\_*\ (settings\_file, num\_processes)
----------------------------------------------------------------

Initialize a Dedupe object with saved settings

Arguments
~~~~~~~~~

``settings_file`` should be the path to settings file produced from the
```writeSettings`` <#wiki-writeSettings>`__ method of a previous, active
Dedupe object.

``num_processes`` should be the number of processes to use for parallel
processing, defaults to 1

Example usage
~~~~~~~~~~~~~

.. code:: python

    deduper = StaticDedupe('my_settings_file')

`threshold, same as Dedupe <#wiki-threshold-dedupe>`__
------------------------------------------------------

`match, same as Dedupe <#wiki-match-dedupe>`__
----------------------------------------------

`blocker, same as Dedupe <#wiki-blocker>`__
-------------------------------------------

`thresholdBlocks, same as Dedupe <#wiki-thresholdBlocks>`__
-----------------------------------------------------------

`matchBlocks, same as Dedupe <#wiki-matchBlocks>`__
---------------------------------------------------

RecordLink
==========

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
-------

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

`Defining a model, same as Dedupe <#wiki-init-dedupe-active>`__
---------------------------------------------------------------

We assume that the fields you want to compare across datasets have the
same field name.

sample(data\_1, data\_2, sample\_size)
--------------------------------------

Draws a random sample of combinations of records from the first and
second datasets, and initializes active learning with this sample

Arguments:
~~~~~~~~~~

``data_1`` is a dictionary of records from first dataset, where the keys
are record\_ids and the values are dictionaries with the keys being
field names.

``data_2`` is dictionary of records from second dataset, same form as
data\_1

``sample_size`` is the size of the sample to draw

Example Usage
~~~~~~~~~~~~~

.. code:: python

    linker.sample(data_1, data_2, 150000)

`uncertainPairs, same as Dedupe <#wiki-uncertainPairs>`__
---------------------------------------------------------

`markPairs, same as Dedupe <#wiki-markPairs>`__
-----------------------------------------------

`train, same as Dedupe <#wiki-train>`__
---------------------------------------

threshold(data\_1, data\_2, recall\_weight)
-------------------------------------------

Returns the threshold that maximizes the expected F score, a weighted
average of precision and recall for a sample of data.

Arguments:
~~~~~~~~~~

``data_1`` is a dictionary of records from first dataset, where the keys
are record\_ids and the values are dictionaries with the keys being
field names

``data_2`` is a dictionary of records from second dataset, same form as
data\_1

``recall_weight`` sets the tradeoff between precision and recall. I.e.
if you care twice as much about recall as you do precision, set
recall\_weight to 2.

Example usage
~~~~~~~~~~~~~

.. code:: python

    > threshold = deduper.threshold(data_1, data_2, recall_weight=2)
    > print threshold
    0.21

 match(data\_1, data\_2, threshold)
-----------------------------------

Identifies pairs of records that refer to the same entity, returns
tuples of record ids, where both record\_ids within a tuple should refer
to the same entity

This method should only used for small to moderately sized datasets for
larger data, use matchBlocks

Arguments:
~~~~~~~~~~

``data_1`` is a dictionary of records from first dataset, where the keys
are record\_ids and the values are dictionaries with the keys being
field names

``data_2`` is a dictionary of records from second dataset, same form as
data\_1

``threshold`` is a number between 0 and 1 (default is .5). We will
consider records as potential duplicates if the predicted probability of
being a duplicate is above the threshold.

Lowering the number will increase recall, raising it will increase
precision

`readTraining, same as Dedupe <#wiki-readTraining>`__
-----------------------------------------------------

`writeSettings, same as Dedupe <#wiki-writeSettings>`__
-------------------------------------------------------

`writeTraining, same as Dedupe <#wiki-writeTraining>`__
-------------------------------------------------------

`blocker, same as Dedupe <#wiki-blocker>`__
-------------------------------------------

`thresholdBlocks, same as Dedupe <#wiki-thresholdBlocks>`__
-----------------------------------------------------------

`matchBlocks, same as Dedupe <#wiki-matchBlocks>`__
---------------------------------------------------

StaticRecordLink
================

Class for record linkage using saved settings. If you have already
trained a record linkage instance, you can load the saved settings with
StaticDedupe.

`Loading settings, same as Dedupe <#wiki-init-dedupe-static>`__
---------------------------------------------------------------

`threshold, same as RecordLink <#wiki-threshold-recordlink>`__
--------------------------------------------------------------

`match, same as RecordLink <#wiki-match-recordlink>`__
------------------------------------------------------

`blocker, same as Dedupe <#wiki-blocker>`__
-------------------------------------------

`thresholdBlocks, same as Dedupe <#wiki-thresholdBlocks>`__
-----------------------------------------------------------

`matchBlocks, same as Dedupe <#wiki-matchBlocks>`__
---------------------------------------------------

Convenience
===========

consoleLable(matcher)
---------------------

Train a matcher instance (Dedupe or RecordLink) from the command line.

Example
-------

.. code:: python

    > dedupe = Dedupe(fields, data_sample)
    > dedupe.consoleLabel(dedupe)

trainingDataLink(data\_1, data\_2, common\_key, training\_size)
---------------------------------------------------------------

Construct training data for consumption by the RecordLink ``markPairs``
method from already linked datasets.

Arguments :
-----------

``data_1`` is a dictionary of records from first dataset, where the keys
are record\_ids and the values are dictionaries with the keys being
field names

``data_2`` is a dictionary of records from second dataset, same form as
data\_1

``common_key`` is the name of the record field that uniquely identifies
a match

``training_size`` the rough limit of the number of training examples,
defaults to 50000

Warning:
~~~~~~~~

Every match must be identified by the sharing of a common key. This
function assumes that if two records do not share a common key then they
are distinct records.

trainingDataDedupe(data, common\_key, training\_size)
-----------------------------------------------------

Construct training data for consumption by the Dedupe ``markPairs``
method from an already deduplicated dataset.

Arguments :
-----------

``data`` is a dictionary of records, where the keys are record\_ids and
the values are dictionaries with the keys being field names

``common_key`` is the name of the record field that uniquely identifies
a match

``training_size`` is the rough limit of the number of training examples,
defaults to 50000

Warning:
~~~~~~~~

Every match must be identified by the sharing of a common key. his
function assumes that if two records do not share a common key then they
are distinct records.
