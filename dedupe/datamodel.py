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
                 'InteractionType' : lambda x : None}

class DataModel(dict) :
    def __init__(self, fields):

        self['bias'] = 0
        self['fields'] = self.buildModel(fields)

        self.fieldDistanceVariables()

        self.total_fields = len(self['fields'])


    def buildModel(self, fields) :
        field_model = []
        interaction_terms = []
        source_variable = None

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
                field_object = field_classes[field_type](definition)
            except KeyError :
                valid_fields = field_classes.keys()
                raise KeyError("Field type %s not valid. Valid types include %s"
                               % (definition['type'], ', '.join(valid_fields)))

                

            if field_object :
                field_model.append(field_object)

            if field_type in ('Categorical', 'Source') :
                field_model.extend(field_object.higher_dummies.values())

                if field_type == 'Source' :
                    source_variable = field_object

            elif field_type == 'Interaction' :
                interaction_terms.append(definition)

        for field, definition in interaction_terms :
            field_model[name] = InteractionType(field, definition, field_model)
            field_model.update(field_model[name].dummyInteractions(field_model))

        for definition in field_model :
            if definition.has_missing :
                field_name = "%s: not missing" % definition.field 
                field_model.append(MissingDataType(field_name))

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





