import json


def _from_json(json_object):
    if '__class__' in json_object:
        if json_object['__class__'] == 'frozenset':
            return frozenset(json_object['__value__'])
        if json_object['__class__'] == 'tuple':
            return tuple(json_object['__value__'])
    return json_object


def hint_tuples(item):
    if isinstance(item, tuple):
        return {'__class__': 'tuple',
                '__value__': list(item)}
    if isinstance(item, list):
        return [hint_tuples(e) for e in item]
    if isinstance(item, dict):
        return {key: hint_tuples(value) for key, value in item.items()}
    else:
        return item


class TupleEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(hint_tuples(obj))

    def iterencode(self, obj):
        return super().iterencode(hint_tuples(obj))

    def default(self, python_object):
        if isinstance(python_object, frozenset):
            return {'__class__': 'frozenset',
                    '__value__': list(python_object)}
        return super().default(python_object)
