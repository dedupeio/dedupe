try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

import itertools
import dedupe.predicates
import dedupe.blocking


from dedupe.fieldclasses import *

field_classes = {'String' : StringType,
                 'ShortString' : ShortStringType,
                 'LatLong' : LatLongType,
                 'Set' : SetType, 
                 'Source' : SourceType,
                 'Text' : TextType,
                 'Categorical' : CategoricalType,
                 'Custom' : CustomType,
                 'InteractionType' : None}

class DataModel(dict) :
    def __init__(self, fields):

        self['bias'] = 0
        self['fields'] = buildModel(fields)
        self['fields'] = interactions(self['fields'])

        self.fieldDistanceVariables()

        self.total_fields = len(self['fields'])

    def interactions(self, field_model) :
        return field_model


    def fieldDistanceVariables(self) :

        fields = self['fields']

        self.interactions = []
        self.categorical_indices = []

        self.field_comparators = OrderedDict([(field.field, 
                                               field.comparator)
                                              for field in fields
                                              if field.comparator])

    
        self.missing_field_indices = [i for i, definition 
                                      in enumerate(fields)
                                      if definition.has_missing]

        for definition in fields :
            if definition.type == 'Interaction' :
                interaction_indices = []
                for interaction_field in definition.interaction_fields :
                    interaction_indices.append(fields.index(interaction_field))
                self.interactions.append(interaction_indices)
            if definition.type in ('Source', 'Categorical') :
                self.categorical_indices.append((field_names.index(field),
                                                 len(definition.higher_dummies)))




def buildModel(fields) :
    field_model = set([])
    interaction_terms = []

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
            valid_fields = field_classes.keys()
            raise KeyError("Field type %s not valid. Valid types include %s"
                           % (definition['type'], ', '.join(valid_fields)))

        if field_type == 'Interaction' :
            interaction_terms.append(definition)
        else :
            field_object = field_class(definition)
            field_model.add(field_object)

            if field_type in ('Categorical', 'Source') :
                field_model.update(field_object.higher_dummies.values())


    return field_model, interaction_terms

