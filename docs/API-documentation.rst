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
        
        matcher = dedupe.Dedupe(variables)

    .. automethod:: prepare_training
    .. automethod:: uncertain_pairs
    .. automethod:: mark_pairs
    .. automethod:: train
    .. automethod:: write_training
    .. automethod:: write_settings
    .. automethod:: cleanup_training

    Matching methods
    
    .. automethod:: partition

    Lower level methods

    .. attribute:: fingerprinter

       Instance of :class:`dedupe.blocking.Fingerprinter` class if
       the :func:`train` has been run, else `None`.

    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: cluster



:class:`StaticDedupe` Objects
-----------------------------
.. autoclass:: dedupe.StaticDedupe

    .. code:: python
    
        with open('learned_settings', 'rb') as f:
            matcher = StaticDedupe(f)

    .. automethod:: partition
    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: cluster
    

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
    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: one_to_one
    .. automethod:: many_to_one		    


:class:`StaticRecordLink` Objects
---------------------------------
.. autoclass:: dedupe.StaticRecordLink

    .. code:: python
    
        with open('learned_settings', 'rb') as f:
            matcher = StaticRecordLink(f)

    .. automethod:: join
    .. automethod:: pairs
    .. automethod:: score
    .. automethod:: one_to_one
    .. automethod:: many_to_one    		    
       

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
    .. automethod:: blocks
    .. automethod:: score
    .. automethod:: many_to_n
       

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


:class:`Fingerprinter` Objects
------------------------------
.. autoclass:: dedupe.blocking.Fingerprinter

   .. automethod:: __call__
   .. automethod:: index
   .. automethod:: unindex	       
   .. automethod:: index_all
   .. automethod:: reset_indices
