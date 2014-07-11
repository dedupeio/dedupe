Field definitions
=================

A field definition describes the records that you want to match. It is
a dictionary where the keys are the fields and the values are the
field specification


.. code:: python

   fields = {
             'Site name': {'type': 'String'},
	     'Address': {'type': 'String'},
	     'Zip': {'type': 'String', 'Has Missing':True},
	     'Phone': {'type': 'String', 'Has Missing':True},
	     }


String Types
^^^^^^^^^^^^

A 'String' type field must have as its key a name of a field as it
appears in the data dictionary and a type declaration ex.
``{'Address': {type: 'String'}}`` The string type expects fields to be of
class string. Missing data should be represented as an empty string
``''``

String types are compared using `affine gap string
distance <http://en.wikipedia.org/wiki/Gap_penalty#Affine>`__.

ShortString Types
^^^^^^^^^^^^^^^^^

Short strings are just like String types except that dedupe will not
try to learn a canopy blocking rule for these fields, which can speed
up the training phase considerably. Zip codes and city names are good
candidates for this type. If in doubt, just use 'String.'

.. code:: python

  {'Zipcode': {type: 'ShortString'}}

.. _text-types-label:

Text Types
^^^^^^^^^^

If you want to compare fields comparing long blocks of text, like
product descriptions or article abstracts you should use this
type. Text types fields are compared using the `cosine similarity
metric <http://en.wikipedia.org/wiki/Vector_space_model>`__.

Basically, this is a measurement of the amount of words that two
documents have in common. This measure can be made more useful the
overlap of rare words counts more than the overlap of common words. If
provide a sequence of example fields than (a corpus), dedupe will
learn these weights for you.

.. code:: python

   {'Product description' : {'type' : 'Text', 
                             'corpus' : ['this product is great',
			                 'this product is great and blue']}
    } 

If you don't want to adjust the measure to your data, just 'corpus'
out of the field definition.

.. code:: python

   {'Product description' : {'type' : 'Text'}
    } 



Custom Types
^^^^^^^^^^^^

A 'Custom' type field must have as its key a name of a field as it
appears in the data dictionary, at 'type' declaration, and a
'comparator' declaration. The comparator must be a function that can
take in two field values and return a number or a numpy.nan when one or both
of the fields is missing.

Example custom comparator:

.. code:: python

  python def sameOrNotComparator(field_1, field_2) :     
    if field_1 and field_2 :         
        if field_1 == field_2 :             
            return 1         
        else:             
            return 0     
    else :         
        return numpy.nan

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

A 'Set' type field is for comparing lists of elements, like keywords
or client names. Set types are very similar to
:ref:`text-types-label`.  They use the same comparison function and
you can also let dedupe learn which terms are common or rare by
providing a corpus. Within a record, a Set types field have to be
hashable sequences like tuples or frozensets.

.. code:: python

    {'Co-authors': {'type': 'Set',
                    'corpus' : [('steve edwards'),
		                ('steve edwards', steve jobs')]}
     } 

or

.. code:: python

    {'Co-authors': {'type': 'Set'}
     } 




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

::

    'Name'          'Business Type' 
    AAA Taxi        taxi 
    AA1 Taxi        taxi 
    Hindelbert Esq  lawyer

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
If one or both fields are missing, the field comparator should return
``numpy.nan.`` By default, dedupe will replace these values with zeros. 

If you want to model this missing data for a field, you can set ``'Has
Missing' : True`` in the field definition. This creates a new,
additional field representing whether the data was present or not and
zeros out the missing data.

If there is missing data, but you did not declare ``'Has
Missing' : True`` then the missing data will simply be zeroed out and
no field will be created to account for missing data.

This approach is called 'response augmented data' and is described in
Benjamin Marlin's thesis `"Missing Data Problems in Machine Learning"
<http://people.cs.umass.edu/~marlin/research/phd_thesis/marlin-phd-thesis.pdf>`__. Basically,
this approach says that, even without looking at the value of the
field comparisons, the pattern of observed and missing responses will
affect the probability that a pair of records are a match.

This approach makes a few assumptions that are usually not completely true:

- Whether a field is missing data is not associated with any other
  field missing data
- That the weighting of the observed differences in field A should be
  the same regardless of whether field B is missing.


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
