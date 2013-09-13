try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.jaccard import compareJaccard


class DataModel(dict) :
    def __init__(self, fields):
        self['bias'] = 0

        self['fields'] = OrderedDict()

        interaction_terms = {}

        for k, v in fields.items():
            self.checkFieldDefinition(v)

            if v['type'] == 'LatLong' :
                v['comparator'] = compareLatLong
            elif v['type'] == 'Set' :
                v['comparator'] = compareJaccard
            elif v['type'] == 'String' :
                v['comparator'] = normalizedAffineGapDistance
            elif v['type'] == 'Interaction' :
                for field in v['Interaction Fields'] :
                    if 'Has Missing' in fields[field] :
                        v.update({'Has Missing' : True})
                        break

                v.update({'weight': 0})
                interaction_terms[k] = v
                # We want the interaction terms to be at the end of of the
                # ordered dict so we'll add them after we finish
                # processing all the other fields
                continue

            self['fields'][k] = v

        self['fields'].update(interaction_terms)

        for k, v in self['fields'].items() :
           if 'Has Missing' in v :
               if v['Has Missing'] :
                   self['fields'][k + ': not_missing'] = {'weight' : 0,
                                                                  'type'   : 'Missing Data'}
           else :
               self['fields'][k].update({'Has Missing' : False})


    def checkFieldDefinition(self, definition) :
        assert definition.__class__ is dict, \
            "Incorrect field specification: field " \
            "specifications are dictionaries that must " \
            "include a type definition, ex. " \
            "{'Phone': {type: 'String'}}"

        assert 'type' in definition, \
            "Missing field type: field " \
            "specifications are dictionaries that must " \
            "include a type definition, ex. " \
            "{'Phone': {type: 'String'}}"

        assert definition['type'] in ['String', 'LatLong', 'Set',
                                      'Custom', 'Interaction'], \
            "Invalid field type: field " \
            "specifications are dictionaries that must " \
            "include a type definition, ex. " \
            "{'Phone': {type: 'String'}}"

        if definition['type'] == 'Custom' :
            assert 'comparator' in v, \
                "For 'Custom' field types you must define " \
                "a 'comparator' function in the field "\
                "definition. "
        else :
            assert 'comparator' not in definition, \
                "Custom comparators can only be " \
                "defined for fields of type 'Custom'"

        if definition['type'] == 'Interaction' :
            assert 'Interaction Fields' in v, \
                'No "Interaction Fields" defined'



