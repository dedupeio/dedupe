=====================
Library Documentation
=====================

:class:`Dedupe` Objects
-----------------------
.. autoclass:: dedupe.Dedupe

    .. code:: python
    
         # initialize from a defined set of fields
         variables = [{'field' : 'Site name', 'type': 'String'},
                      {'field' : 'Address', 'type': 'String'},
                      {'field' : 'Zip', 'type': 'String', 'has missing':True},
                      {'field' : 'Phone', 'type': 'String', 'has missing':True}
                      ]
         
         deduper = dedupe.Dedupe(variables)
	       
    .. automethod:: prepare_training
    .. automethod:: uncertain_pairs
    .. automethod:: mark_pairs
    .. automethod:: train
    .. automethod:: write_training
    .. automethod:: write_settings
    .. automethod:: cleanup_training
    .. automethod:: partition



:class:`StaticDedupe` Objects
-----------------------------
.. autoclass:: dedupe.StaticDedupe

    .. code:: python
    
        with open('learned_settings', 'rb') as f:
            matcher = StaticDedupe(f)

    .. automethod:: partition
    

:class:`RecordLink` Objects
---------------------------
.. autoclass:: dedupe.RecordLink

    .. code:: python
     
        # initialize from a defined set of fields
        variables = [{'field' : 'Site name', 'type': 'String'},
                     {'field' : 'Address', 'type': 'String'},
                     {'field' : 'Zip', 'type': 'String', 'has missing':True},
                     {'field' : 'Phone', 'type': 'String', 'has missing':True}
                     ]
        
        deduper = dedupe.RecordLink(variables)

    .. automethod:: prepare_training
    .. automethod:: uncertain_pairs
    .. automethod:: mark_pairs
    .. automethod:: train
    .. automethod:: write_training
    .. automethod:: write_settings
    .. automethod:: cleanup_training
    .. automethod:: join


:class:`StaticRecordLink` Objects
---------------------------------
.. autoclass:: dedupe.StaticRecordLink

    .. code:: python
    
        with open('learned_settings', 'rb') as f:
            matcher = StaticRecordLink(f)

    .. automethod:: join
       

:class:`Gazetteer` Objects
--------------------------
.. autoclass:: dedupe.Gazetteer

    .. code:: python
     
        # initialize from a defined set of fields
        variables = [{'field' : 'Site name', 'type': 'String'},
                     {'field' : 'Address', 'type': 'String'},
                     {'field' : 'Zip', 'type': 'String', 'has missing':True},
                     {'field' : 'Phone', 'type': 'String', 'has missing':True}
                     ]
        
        matcher = dedupe.Gazetteer(variables)

    .. automethod:: prepare_training
    .. automethod:: uncertain_pairs
    .. automethod:: mark_pairs
    .. automethod:: train
    .. automethod:: write_training
    .. automethod:: write_settings
    .. automethod:: cleanup_training
    .. automethod:: index
    .. automethod:: unindex
    .. automethod:: search
       

:class:`StaticGazetteer` Objects
--------------------------------
.. autoclass:: dedupe.StaticGazetteer

    .. code:: python
    
        with open('learned_settings', 'rb') as f:
            matcher = StaticGazetteer(f)

    .. automethod:: index
    .. automethod:: unindex
    .. automethod:: search
    .. automethod:: blocks
    .. automethod:: score
    .. automethod:: many_to_n

Lower Level Classes and Methods
-------------------------------

With the methods documented above, you can work with data into the
millions of records. However, if are working with larger data you
may not be able to load all your data into memory. You'll need
to interact with some of the lower level classes and methods.

.. seealso:: The `PostgreSQL <https://dedupeio.github.io/dedupe-examples/docs/pgsql_big_dedupe_example.html>`_ and `MySQL <https://dedupeio.github.io/dedupe-examples/docs/mysql_example.html>`_ examples use these lower level classes and methods.

Dedupe and StaticDedupe
***********************

.. currentmodule:: dedupe

.. class:: Dedupe
   :noindex:

    .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class if
       the :func:`train` has been run, else `None`.
    
    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: cluster

.. class:: StaticDedupe
   :noindex:

    .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class
    
    .. method:: pairs(data)

       Same as :func:`dedupe.Dedupe.pairs`
		
    .. method:: score(pairs)

       Same as :func:`dedupe.Dedupe.score`

    .. method:: cluster(scores, threshold=0.5)

       Same as :func:`dedupe.Dedupe.cluster`
		    
    
RecordLink and StaticRecordLink
*******************************

.. class:: RecordLink
   :noindex:

    .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class if
       the :func:`train` has been run, else `None`.

    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: one_to_one
    .. automethod:: many_to_one		    

.. class:: StaticRecordLink
   :noindex:

   .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class

   .. method:: pairs(data_1, data_2)

	Same as :func:`dedupe.RecordLink.pairs`

   .. method:: score(pairs)

	Same as :func:`dedupe.RecordLink.score`

   .. method:: one_to_one(scores, threshold=0.0)

        Same as :func:`dedupe.RecordLink.one_to_one`

   .. method:: many_to_one(scores, threshold=0.0)

	Same as :func:`dedupe.RecordLink.many_to_one`


Gazetteer and StaticGazetteer
*****************************

.. class:: Gazetteer
   :noindex:

    .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class if
       the :func:`train` has been run, else `None`.

    .. automethod:: blocks
    .. automethod:: score
    .. automethod:: many_to_n

.. class:: StaticGazeteer
   :noindex:

    .. attribute:: fingerprinter
    
       Instance of :class:`dedupe.blocking.Fingerprinter` class
	   
    .. method:: blocks(data)

	Same as :func:`dedupe.Gazetteer.blocks`

    .. method:: score(blocks)

	Same as :func:`dedupe.Gazetteer.score`

    .. method:: many_to_n(score_blocks, threshold=0.0, n_matches=1)

	Same as :func:`dedupe.Gazetteer.many_to_n`		
		    
:class:`Fingerprinter` Objects
******************************
.. autoclass:: dedupe.blocking.Fingerprinter

   .. automethod:: __call__
   .. autoattribute:: index_fields
   .. automethod:: index
   .. automethod:: unindex	       
   .. automethod:: reset_indices


Convenience Functions
---------------------

.. autofunction:: dedupe.console_label
.. autofunction:: dedupe.training_data_dedupe
.. autofunction:: dedupe.training_data_link
.. autofunction:: dedupe.canonicalize
