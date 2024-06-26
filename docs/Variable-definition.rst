.. _variable_definitions:

Variable Definitions
====================

Variables
---------

A variable definition describes the records that you want to match. It is sequence
of Variable objects. For example:-

.. code:: python

    import dedupe.variables
	  
    [
        dedupe.variables.String("Site Name"),
        dedupe.variables.String("Address"),
	dedupe.variables.ShortString("Zip", has_missing=True),
	dedupe.variables.String("Phone", has_missing=True)
    ]


String
^^^^^^

The ``String`` takes the key of the record field to compare.

``String`` variables are compared using string edit distance, specifically
`affine gap string distance <http://en.wikipedia.org/wiki/Gap_penalty#Affine>`__.
This is a good metric for measuring fields that might have typos in them,
such as "John" vs "Jon".

For example:-

.. code:: python

  dedupe.variables.String("Address")

ShortString
^^^^^^^^^^^

The ``ShortString`` variable is just like the ``String`` variable except that dedupe
will not try to learn any :ref:`index blocking rules <index-blocks-label>` for these fields, which can
speed up the training phase considerably.

Zip codes and city names are good candidates for this variable. If in doubt,
always use ``String``.

For example:-

.. code:: python

  dedupe.variables.ShortString("Zipcode")

.. _text-types-label:

Text
^^^^

If you want to compare fields containing blocks of text e.g. product
descriptions or article abstracts, you should use this variable. ``Text``
variables are compared using the `cosine similarity metric
<http://en.wikipedia.org/wiki/Vector_space_model>`__.

This is a measurement of the amount of words that two documents have in
common. This measure can be made more useful as the overlap of rare words
counts more than the overlap of common words.

Compare this to ``String`` and ``ShortString`` variables: For strings
containing occupations, "yoga teacher" might be fairly similar to
"yoga instructor" when using the ``Text`` measurement, because they
both contain the relatively rare word of "yoga". However, if you
compared these two strings using the ``String`` or ``ShortString``
measurements, they might be considered fairly dissimilar, because the
actual string edit distance between them is large.


If provided a sequence of example fields (i.e. a corpus) then dedupe will
learn these weights for you. For example:-

.. code:: python

   dedupe.variables.Text("Product description",
                         corpus=[
                                 'this product is great',
                                 'this product is great and blue'
                                ]
			)

If you don't want to adjust the measure to your data, just leave 'corpus' out
of the variable definition entirely.

.. code:: python

   dedupe.variables.Text("Product description") 


Custom Variable
^^^^^^^^^^^^

A ``Custom`` variables allwos you to use a custom function for
comparing fields. The function must take two field values and return a
number.

For example, a custom comparator:

.. code:: python

  def same_or_not_comparator(field_1, field_2):     
    if field_1 and field_2 :         
        if field_1 == field_2 :             
            return 0         
        else:             
            return 1     

The corresponding variable definition:

.. code:: python

    dedupe.variables.Custom("Zip", comparator=same_or_not_comparator)

``Custom`` variables do not have any blocking rules associated with them.
Since dedupe needs blocking rules, a data model that only contains ``Custom``
fields will raise an error.

LatLong
^^^^^^^

A ``LatLong`` variables are compared using the `Haversine
Formula <http://en.wikipedia.org/wiki/Haversine_formula>`__. 

A ``LatLong`` variable field must consist of tuples of floats
corresponding to a latitude and a longitude.

.. code:: python

    dedupe.variables.LatLong("location")

Set
^^^

A ``Set`` variables are for comparing lists of elements, like keywords or
client names. ``Set`` variables are very similar to :ref:`text-types-label`. They
use the same comparison function and you can also let dedupe learn which
terms are common or rare by providing a corpus. Within a record, a ``Set``
variable field has to be hashable sequences like tuples or frozensets.

.. code:: python

    dedupe.variables.Set("Co-authors",
                         corpus=[
                                 ('steve edwards'),
                                 ('steve edwards', 'steve jobs')
                                ])

or

.. code:: python

    dedupe.variables.Set("Co-authors")

Interaction
^^^^^^^^^^^

An ``Interaction`` variable multiplies the values of the multiple variables.
The arguments to an ``Interaction`` variable must be a sequence of variable names of
other fields you have defined in your variable definition.

`Interactions <http://en.wikipedia.org/wiki/Interaction_%28statistics%29>`__
are good when the effect of two predictors is not simply additive.

.. code:: python

    [
        dedupe.variables.String("Name", name="name"),
	dedupe.variables.Custom("Zip", comparator=same_or_not_comparator, name="zip")
	dedupe.variables.Interaction("name", "zip")
    ]

Exact
^^^^^

``Exact`` variables measure whether two fields are exactly the same or not.

.. code:: python

    dedupe.variables.Exact("city")


Exists
^^^^^^

``Exists`` variables are useful if the presence or absence of a field tells you
something meaningful about a pair of records. It differentiates between three
different cases:

    1. The field is missing in both records.
    2. The field is missing in one of the records.
    3. The field is present in neither of the records.

.. code:: python

    dedupe.variables.Exists("first_name")



Categorical
^^^^^^^^^^^

``Categorical`` variables are useful when you are dealing with qualitatively
different types of things. For example, you may have data on businesses and
you find that taxi cab businesses tend to have very similar names but law
firms don't. ``Categorical`` variables would let you indicate whether two records
are both taxi companies, both law firms, or one of each. This is also a good choice
for fields that are booleans, e.g. "True" or "False".

Dedupe would represent these three possibilities using two dummy variables:

::

    taxi-taxi      0 0
    lawyer-lawyer  1 0
    taxi-lawyer    0 1

A categorical field declaration must include a list of all the different
strings that you want to treat as different categories.

So if you data looks like this:-

::

    'Name'          'Business Type' 
    AAA Taxi        taxi 
    AA1 Taxi        taxi 
    Hindelbert Esq  lawyer

You would create a definition such as:

.. code:: python

    dedupe.variables.Categorical("Business Type", categories=['taxi', 'lawyer'])

Price
^^^^^

``Price`` variables are useful for comparing positive, non-zero numbers like
prices. The values of ``Price`` field must be a positive float. If the value is
0 or negative, then an exception will be raised.

.. code:: python

    dedupe.variables.Price("cost")

Optional Variables
------------------

These variables aren't included in the core of dedupe, but are available to
install separately if you want to use them.

In addition to the several variables below, you can find `more optional
variables on GitHub <https://github.com/search?q=org%3Adedupeio+dedupe-variable>`__.  

DateTime
^^^^^^^^

``DateTime`` variables are useful for comparing dates and timestamps. This
variable can accept strings or Python datetime objects as inputs.

The ``DateTime`` variable a few optional arguments that can help
improve behavior if you know your field follows an unusual format:

* :code:`fuzzy` - Use fuzzy parsing to automatically extract dates from strings like "It happened on June 2nd, 2018" (default :code:`True`)
* :code:`dayfirst` - Ambiguous dates should be parsed as dd/mm/yy (default :code:`False`)
* :code:`yearfirst`-  Ambiguous dates should be parsed as yy/mm/dd (default :code:`False`)

Note that the ``DateTime`` variable defaults to mm/dd/yy for ambiguous dates.
If both :code:`dayfirst` and :code:`yearfirst` are set to :code:`True`, then
:code:`dayfirst` will take precedence.


.. code:: python

    import datetimetype

    datetimetype.DateTime("field")

To install:

.. code:: console

    pip install dedupe-variable-datetime


Address
^^^^^^^

An ``USAddress`` variable should be used for United States addresses. It uses
the `usaddress <https://usaddress.readthedocs.io/en/latest/>`__ package to
split apart an address string into components like address number, street
name, and street type and compares component to component.

For example:-

.. code:: python

    import addressvariable
	  
    addressvariable.USAddress("address")


To install:

.. code:: console

    pip install dedupe-variable-address


Name
^^^^

A ``WesternName`` variable should be used for a field that contains American names,
corporations and households. It uses the `probablepeople
<https://probablepeople.readthedocs.io/en/latest/>`__ package to split apart
an name string into components like give name, surname, generational suffix,
for people names, and abbreviation, company type, and legal form for
corporations.

For example:-

.. code:: python

    import namevariable

    namevariable.WesternName("field")

.. code:: console

    pip install dedupe-variable-name


Missing Data 
------------ 
If the value of field is missing, that missing value should be represented as 
a ``None`` object. You should also use ``None`` to represent empty strings
(eg ``''``).

.. code:: python

   [
        {'Name': 'AA Taxi', 'Phone': '773.555.1124'},
        {'Name': 'AA Taxi', 'Phone': None},
        {'Name': None, 'Phone': '773-555-1123'}
   ]

If you want to model this missing data for a field, you can set the ``has
missing=True`` in the variable definition. This creates a new,
additional field representing whether the data was present or not and
zeros out the missing data.

If there is missing data, but you did not declare ``has
missing=True`` then the missing data will simply be zeroed out and
no field will be created to account for missing data.

This approach is called 'response augmented data' and is described in
Benjamin Marlin's thesis `"Missing Data Problems in Machine Learning"
<http://people.cs.umass.edu/~marlin/research/phd_thesis/marlin-phd-thesis.pdf>`__.
Basically, this approach says that, even without looking at the value of the
field comparisons, the pattern of observed and missing responses will affect
the probability that a pair of records are a match.

This approach makes a few assumptions that are usually not completely true:

- Whether a field is missing data is not associated with any other field missing data.
- That the weighting of the observed differences in field A should be the same regardless of whether field B is missing.


If you define an an interaction with a field that you declared to have
missing data, then ``has missing=True`` will also be set for the
Interaction field.

Longer example of a variable definition:

.. code:: python

    [
        dedupe.variables.String("name", name="name"),
	dedupe.variables.String("address"),
	dedupe.variables.String("city", name="city"),
	dedupe.variables.Custom("zip", comparator=same_or_not_comparator),
	dedupe.variables.String("cuisine", has_missing=True),
	dedupe.vairables.Interaction("name", "city")
    ]

Multiple Variables comparing same field
--------------------------------------- 
It is possible to define multiple variables that all compare the same
variable.

For example:-

.. code:: python

    [
        dedupe.variables.String("name"),
	dedupe.variables.Text("name")
    ]


Will create two variables that both compare the 'name' field but 
in different ways.


Optional Edit Distance
----------------------

For ``String``, ``ShortString``, ``Address``, and ``Name`` fields, you can
choose to use the a conditional random field distance measure for strings.
This measure can give you more accurate results but is much slower than the
default edit distance.

.. code:: python

    dedupe.variables.String("name", crf=True)
