try:
    from json.scanner import py_make_scanner
    import json
except ImportError:
    from simplejson.scanner import py_make_scanner
    import simplejson as json

import dedupe.core
    


def _to_json(python_object):                                             
    if isinstance(python_object, frozenset):                                
        return {'__class__': 'frozenset',
                '__value__': list(python_object)}
    if isinstance(python_object, dedupe.core.frozendict) :
        return dict(python_object)

    raise TypeError(repr(python_object) + ' is not JSON serializable') 

def _from_json(json_object):                                   
    if '__class__' in json_object:                            
        if json_object['__class__'] == 'frozenset':
            return frozenset(json_object['__value__'])
    return json_object

class dedupe_decoder(json.JSONDecoder):

    def __init__(self, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=_from_json, **kwargs)
        # Use the custom JSONArray
        self.parse_array = self.JSONArray
        # Use the python implemenation of the scanner
        self.scan_once = py_make_scanner(self) 

    def JSONArray(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        return tuple(values), end
