try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

import itertools
import dedupe.predicates
import dedupe.blocking

from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.categorical import CategoricalComparator

valid_fields = ['String',
                'ShortString',
                'LatLong',
                'Set',
                'Source',
                'Text',
                'Categorical',
                'Custom',
                'Interaction']

class DataModel(dict) :
    def __init__(self, fields):

        self['bias'] = 0
        self['fields'] = self.buildModel(fields)

        self.fieldDistanceVariables()

        self.total_fields = len(self['fields'])

    def buildModel(self, fields) :
        field_model = OrderedDict()
        interaction_terms = OrderedDict()
        source_interactions = OrderedDict()
        categoricals = OrderedDict()

        for field, definition in fields.iteritems():

            self.checkFieldDefinitions(definition)

            if definition['type'] == 'LatLong' :
                field_model[field] = LatLongType(field, definition)
                
                
            elif definition['type'] == 'String' :
                field_model[field] = StringType(field, definition)

            elif definition['type'] == 'Set' :
                if 'corpus' not in definition :
                    definition['corpus'] = None 
                field_model[field] = SetType(field, definition)

            elif definition['type'] == 'Text' :
                if 'corpus' not in definition :
                    definition['corpus'] = None 
                field_model[field] = TextType(field, definition)

            elif definition['type'] == 'ShortString' :
                field_model[field] = ShortStringType(field, definition)

            elif definition['type'] == 'Custom' :
                field_model[field] = CustomType(field, definition)
            
            elif definition['type'] == 'Categorical' :
                field_model[field] = CategoricalType(field, definition) 
                categoricals.update(field_model[field].higher_dummies)

            elif definition['type'] == 'Source' :
                field_model[field] = SourceType(field, definition) 
                categoricals.update(field_model[field].higher_dummies)

                for other_field in fields :
                    if other_field != field :
                        source_interactions[str((field, other_field))] =\
                            {"type" : "Interaction",
                             "Interaction Fields" : (field, other_field)}

            elif definition['type'] == 'Interaction' :
                interaction_terms[field] = definition

        field_model = OrderedDict(field_model.items()
                                  + categoricals.items())

        interaction_terms.update(source_interactions)

        for field, definition in interaction_terms.items() :
            field_model[field] = InteractionType(field, definition, field_model)
            field_model.update(field_model[field].dummyInteractions(field_model))

        for field, definition in  field_model.items() :
            if definition.has_missing :
                field_name = "%s: not missing" % field 
                field_model[field_name] = MissingDataType(field_name)

        return field_model

    def checkFieldDefinitions(self, definition) :
        if definition.__class__ is not dict:
            raise ValueError("Incorrect field specification: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )

        elif 'type' not in definition:
            raise ValueError("Missing field type: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )

        elif definition['type'] not in valid_fields :
            raise ValueError("Field type %s not valid. Valid types include %s"
                             % (definition['type'], ', '.join(valid_fields)))
        
        elif definition['type'] != 'Custom' and 'comparator' in definition :
            raise ValueError("Custom comparators can only be defined "
                             "for fields of type 'Custom'")
                
        elif definition['type'] == 'Custom' and 'comparator' not in definition :
                raise ValueError("For 'Custom' field types you must define "
                                 "a 'comparator' function in the field "
                                 "definition. ")


    def fieldDistanceVariables(self) :

        fields = self['fields']
        field_names = fields.keys()

        self.interactions = []
        self.categorical_indices = []

        self.field_comparators = OrderedDict([(field, fields[field].comparator)
                                              for field in fields
                                              if fields[field].comparator])

    
        self.missing_field_indices = [i for i, (field, definition) 
                                      in enumerate(fields.items())
                                      if definition.has_missing]

        for field, definition in fields.items() :
            if definition.type == 'Interaction' :
                interaction_indices = []
                for interaction_field in definition.interaction_fields :
                    interaction_indices.append(field_names.index(interaction_field))
                self.interactions.append(interaction_indices)
            if definition.type in ('Source', 'Categorical') :
                self.categorical_indices.append((field_names.index(field),
                                                 len(definition.higher_dummies)))



class FieldType(object) :
    weight = 0
    comparator = None
    _predicate_functions = []    
             
    def __init__(self, field, definition) :
        self.field = field

        if definition.get('Has Missing', False) :
            self.has_missing = True
        else :
            self.has_missing = False

        self.predicates = [dedupe.blocking.SimplePredicate(pred, field) 
                           for pred in self._predicate_functions]

class ShortStringType(FieldType) :
    comparator = normalizedAffineGapDistance
    type = "ShortString"

    _predicate_functions = (dedupe.predicates.wholeFieldPredicate,
                            dedupe.predicates.tokenFieldPredicate,
                            dedupe.predicates.firstTokenPredicate,
                            dedupe.predicates.commonIntegerPredicate,
                            dedupe.predicates.nearIntegersPredicate,
                            dedupe.predicates.firstIntegerPredicate,
                            dedupe.predicates.sameThreeCharStartPredicate,
                            dedupe.predicates.sameFiveCharStartPredicate,
                            dedupe.predicates.sameSevenCharStartPredicate,
                            dedupe.predicates.commonFourGram,
                            dedupe.predicates.commonSixGram)


class StringType(ShortStringType) :
    comparator = normalizedAffineGapDistance
    type = "String"

    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, field, definition) :
        super(StringType, self).__init__(field, definition)

        canopy_predicates = [dedupe.blocking.TfidfPredicate(threshold, field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates


class TextType(StringType) :
    type = "Text"

    def __init__(self, field, definition) :
        super(TextType, self).__init__(field, definition)

        self.comparator = dedupe.distance.CosineTextSimilarity(definition['corpus'])


class LatLongType(FieldType) :
    comparator = compareLatLong
    type = "LatLong"

    _predicate_functions = [dedupe.predicates.latLongGridPredicate]


class SetType(FieldType) :
    type = "Set"

    _predicate_functions = (dedupe.predicates.wholeSetPredicate,
                         dedupe.predicates.commonSetElementPredicate)

    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, field, definition) :
        super(SetType, self).__init__(field, definition)

        canopy_predicates = [dedupe.blocking.TfidfSetPredicate(threshold, field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

        self.comparator = dedupe.distance.CosineSetSimilarity(definition['corpus'])


class HigherDummyType(FieldType) :
    type = "HigherOrderDummy"

    def __init__(self, field, definition) :
        super(HigherDummyType, self ).__init__(field, definition)

        self.value = definition["value"]
        
class CategoricalType(FieldType) :
    type = "Categorical"
    

    def _categories(self, definition) :
        try :
            categories = definition["Categories"]
        except KeyError :
            raise ValueError('No "Categories" defined')
        
        return categories

    def __init__(self, field, definition) :

        super(CategoricalType, self ).__init__(field, definition)
        
        categories = self._categories(definition)

        self.comparator = CategoricalComparator(categories)

        self.higher_dummies = OrderedDict()

        for value, combo in sorted(self.comparator.combinations[2:]) :
            combo = str(combo)
            self.higher_dummies[str(combo)] = HigherDummyType(combo, 
                                                              {'value' : value,
                                                               'Has Missing' : self.has_missing})

class SourceType(CategoricalType) :
    type = "Source"

    def _categories(self, definition) :
        try :
            categories = definition["Source Names"]
        except KeyError :
            raise ValueError('No "Source Names" defined')

        if len(categories) != 2 :
            raise ValueError("You must supply two and only " 
                             "two source names")
        
        return categories            


class InteractionType(FieldType) :
    type = "Interaction"
    
    def __init__(self, field_name, definition, field_model) :
        super(InteractionType, self).__init__(field_name, definition)

        interactions = definition["Interaction Fields"]
        self.interaction_fields = self.atomicInteractions(interactions,
                                                          field_model)
        for field in self.interaction_fields :
            if field_model[field].has_missing :
                self.has_missing = True

    def atomicInteractions(self, interactions, field_model) :
        atomic_interactions = []
        
        for field in interactions :
            if field_model[field].type == "Interaction" :
                sub_interactions = field_model[field].interaction_fields
                atomic_interactions.extend(self.atomicInteractions(sub_interactions,
                                                                   field_model))
            else :
                atomic_interactions.append(field)

        return atomic_interactions


    def dummyInteractions(self, field_model) :
        dummy_interactions = OrderedDict()

        categorical_fields = set([])
        higher_order_dummies = []

        for field in self.interaction_fields :
            if field_model[field].type in ('Categorical', 'Source') :
                categorical_fields.add(field)
                dummies = [field]
                dummies.extend(field_model[field].higher_dummies)
                higher_order_dummies.append(dummies)
        
        other_fields = set(self.interaction_fields) - categorical_fields
        other_fields = tuple(other_fields)

        for level in itertools.product(*higher_order_dummies) :
            if level and set(level) != categorical_fields :
                interaction_fields = level + other_fields
                field = str(interaction_fields)
                dummy_interactions[field] =\
                    InteractionType(field,
                                    {"Interaction Fields" : interaction_fields,
                                     "Has Missing" : self.has_missing},
                                    field_model)

        return dummy_interactions

class MissingDataType(FieldType) :
    type = "MissingData"

    def __init__(self, field) :
        super(MissingDataType, self).__init__(field, {})
    

class CustomType(FieldType) :
    type = "Custom"

    def __init__(self, field, definition) :
        super(CustomType, self).__init__(field, definition)

        self.comparator = definition["comparator"]


