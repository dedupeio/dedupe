=====================
Library Documentation
=====================

:class:`Dedupe` Objects
-----------------------
.. autoclass:: dedupe.Dedupe
    :members:
    :inherited-members:

    .. code:: python
     
        # initialize from a defined set of fields
        variables = [{'field' : 'Site name', 'type': 'String'},
                     {'field' : 'Address', 'type': 'String'},
                     {'field' : 'Zip', 'type': 'String', 'has missing':True},
                     {'field' : 'Phone', 'type': 'String', 'has missing':True}
                     ]
        
        matcher = dedupe.Dedupe(variables)



:class:`StaticDedupe` Objects
-----------------------------
.. autoclass:: dedupe.StaticDedupe
    :members:
    :inherited-members:

    .. code:: python
    
        with open('my_settings_file', 'rb') as f:
            matcher = StaticDedupe(f)
       

:class:`RecordLink` Objects
---------------------------
.. autoclass:: dedupe.RecordLink
    :members:
    :inherited-members:

    .. code:: python
     
        # initialize from a defined set of fields
        variables = [{'field' : 'Site name', 'type': 'String'},
                     {'field' : 'Address', 'type': 'String'},
                     {'field' : 'Zip', 'type': 'String', 'has missing':True},
                     {'field' : 'Phone', 'type': 'String', 'has missing':True}
                     ]
        
        deduper = dedupe.RecordLink(variables)


:class:`StaticRecordLink` Objects
---------------------------------
.. autoclass:: dedupe.StaticRecordLink
    :members:
    :inherited-members:

:class:`Gazetteer` Objects
--------------------------
.. autoclass:: dedupe.Gazetteer
    :members:
    :inherited-members:

:class:`StaticGazetteer` Objects
--------------------------------
.. autoclass:: dedupe.StaticGazetteer
    :members:
    :inherited-members:

