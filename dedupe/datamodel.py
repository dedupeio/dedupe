try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

import itertools
import dedupe.predicates
import dedupe.blocking

import dedupe.fieldclasses
from dedupe.fieldclasses import MissingDataType

field_classes = {'String' : dedupe.fieldclasses.StringType,
                 'ShortString' : dedupe.fieldclasses.ShortStringType,
                 'LatLong' : dedupe.fieldclasses.LatLongType,
                 'Set' : dedupe.fieldclasses.SetType, 
                 'Text' : dedupe.fieldclasses.TextType,
                 'Categorical' : dedupe.fieldclasses.CategoricalType,
                 'Exists' : dedupe.fieldclasses.ExistsType,
                 'Custom' : dedupe.fieldclasses.CustomType,
                 'Exact' : dedupe.fieldclasses.ExactType,
                 'Interaction' : dedupe.fieldclasses.InteractionType}

class DataModel(dict) :
    def __len__(self) :
        return len(self['fields'])

    def __init__(self, fields):

        self['bias'] = 0

        field_model = typifyFields(fields)
        self.interaction_start = len(field_model)
        print field_model

        field_model = interactions(field_model)
        field_model = missing(field_model)

        self['fields'] = field_model

        self.field_comparators, self.derived_start = self._fieldComparators()

    def _fieldComparators(self) :
        start = 0
        stop = 0 
        comparators = []
        for field in self['fields'] :
            if hasattr(field, 'comparator') :
                stop = start + len(field) 
                comparators.append((field.field, field.comparator, start, stop))
                start = stop

        return comparators, stop


    @property 
    def missing_field_indices(self) : 
        return [i for i, definition 
                in enumerate(self['fields'])
                if definition.has_missing]

    @property
    def interactions(self) :
        indices = []

        fields = self['fields']
        field_names = [field.name for field in fields]

        for definition in fields :
            if hasattr(definition, 'interaction_fields') :
                interaction_indices = []
                for interaction_field in definition.interaction_fields :
                    interaction_indices.append(field_names.index(interaction_field))
                indices.append(interaction_indices)
                
        return indices

def typifyFields(fields) :
    field_model = []

    for definition in fields :
        try :
            field_type = definition['type']
        except TypeError :
            raise TypeError("Incorrect field specification: field "
                            "specifications are dictionaries that must "
                            "include a type definition, ex. "
                            "{'field' : 'Phone', type: 'String'}")
        except KeyError :
            raise KeyError("Missing field type: fields "
                           "specifications are dictionaries that must "
                           "include a type definition, ex. "
                           "{'field' : 'Phone', type: 'String'}")
        try :
            field_class = field_classes[field_type]
        except KeyError :
            raise KeyError("Field type %s not valid. Valid types include %s"
                           % (definition['type'], ', '.join(field_classes)))

        field_object = field_class(definition)
        field_model.append(field_object)
        
        if hasattr(field_object, 'higher_dummies') :
            field_model.extend(field_object.higher_dummies)

    return field_model

def missing(field_model) :
    for definition in field_model[:] :
        if definition.has_missing :
            field_model.append(MissingDataType(definition.name))

    return field_model

def interactions(field_model) :
    field_d = dict((field.name, field) for field in field_model)

    for field in field_model[:] : 
        if hasattr(field, 'expandInteractions') :
            field.expandInteractions(field_d)
            field_model.extend(field.higher_vars)

    return field_model

