try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

import itertools
import dedupe.predicates
import dedupe.blocking

import dedupe.fieldclasses
from dedupe.fieldclasses import InteractionType, MissingDataType

field_classes = {'String' : dedupe.fieldclasses.StringType,
                 'ShortString' : dedupe.fieldclasses.ShortStringType,
                 'LatLong' : dedupe.fieldclasses.LatLongType,
                 'Set' : dedupe.fieldclasses.SetType, 
                 'Source' : dedupe.fieldclasses.SourceType,
                 'Text' : dedupe.fieldclasses.TextType,
                 'Categorical' : dedupe.fieldclasses.CategoricalType,
                 'Custom' : dedupe.fieldclasses.CustomType,
                 'Exact' : dedupe.fieldclasses.ExactType,
                 'Interaction' : dedupe.fieldclasses.InteractionType}

class DataModel(dict) :
    def __init__(self, fields):

        self['bias'] = 0

        field_model = typifyFields(fields)

        field_model = sourceFields(field_model)
        field_model = interactions(field_model)
        field_model = missing(field_model)

        field_model = sorted(field_model)

        self['fields'] = field_model

        self.total_fields = len(self['fields'])


    @property
    def field_comparators(self) :
        return [(field.field, field.comparator) 
                for field 
                in self['fields'] 
                if hasattr(field, 'comparator')]

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

    @property
    def categorical_indices(self) :

        indices = []
        field_model = self['fields']

        for definition in self['fields'] :
            if hasattr(definition, 'dummies') :
                indices.append((field_model.index(definition),
                                len(definition.dummies)))

        return indices

def typifyFields(fields) :
    field_model = set([])

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
        field_model.add(field_object)
        
        if hasattr(field_object, 'dummies') :
            field_model.update(field_object.dummies)

    return field_model

def missing(field_model) :
    for definition in list(field_model) :
        if definition.has_missing :
            field_model.add(MissingDataType(definition.name))

    return field_model

def interactions(field_model) :
    field_d = dict((field.name, field) for field in field_model)

    for field in list(field_model) : 
        if hasattr(field, 'expandInteractions') :
            field.expandInteractions(field_d)
            field_model.update(field.dummyInteractions(field_d))

    return field_model

def sourceFields(field_model) :
    source_fields = [field for field in field_model 
                     if field.type == "Source"]

    for source_field in source_fields :
        for field in list(field_model) :
            if field != source_field :
                if (not hasattr(field, 'base_name') 
                    or field.base_name != source_field.name) :
                    interaction = InteractionType({"interaction variables" : 
                                                   (source_field.name, 
                                                    field.name)})
                    field_model.add(interaction)

    return field_model
