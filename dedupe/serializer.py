import simplejson as json
import dedupe.core
import collections

def _from_json(json_object):                                   
    if '__class__' in json_object:                            
        if json_object['__class__'] == 'frozenset':
            return frozenset(json_object['__value__'])
        if json_object['__class__'] == 'tuple':
            return tuple(json_object['__value__'])
    return json_object

def _to_json(python_object) :
    if isinstance(python_object, frozenset):                                
        python_object = {'__class__': 'frozenset',
                         '__value__': list(python_object)}
    elif isinstance(python_object, tuple) :
        python_object = {'__class__': 'tuple',
                         '__value__': list(python_object)}
    elif isinstance(python_object, dedupe.core.frozendict) :
        python_object = dict(python_object)
    else :
        raise TypeError(repr(python_object) + ' is not JSON serializable') 

    return python_object


class dedupe_decoder(json.JSONDecoder):

    def __init__(self, **kwargs):
        json._toggle_speedups(False) # in simplejson, without this
                                     # some strings can be bytestrings
                                     # instead of unicode
                                     # https://code.google.com/p/simplejson/issues/detail?id=40
        json.JSONDecoder.__init__(self, object_hook=_from_json, **kwargs)


