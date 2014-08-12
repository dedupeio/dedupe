import itertools
import dedupe
from collections import defaultdict

from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.categorical import CategoricalComparator

try:
    from collections import OrderedDict
except ImportError :
    from dedupe.backport import OrderedDict

class Variable(object) :
    def __lt__(self, other) :
        return self.sort_level < other.sort_level

    def __repr__(self) :
        return self.name

    def __hash__(self) :
        return hash(self.name)

    def __eq__(self, other) :
        return self.name == other.name

    def __init__(self, definition) :

        self.weight = 0

        if definition.get('has missing', False) :
            self.has_missing = True
        else :
            self.has_missing = False

class FieldType(Variable) :
    sort_level = 0
             
    def __init__(self, definition) :
        self.field = definition['field']

        if 'variable name' in definition :
            self.name = definition['variable name'] 
        else :
            self.name = "(%s: %s)" % (self.field, self.type)

        self.predicates = [dedupe.blocking.SimplePredicate(pred, self.field) 
                           for pred in self._predicate_functions]

        super(FieldType, self).__init__(definition)

class ExactType(FieldType) :
    _predicate_functions = [dedupe.predicates.wholeFieldPredicate]
    type = "Exact"

    @staticmethod
    def comparator(field_1, field_2) :
        if field_1 == field_2 :
            return 0
        else :
            return 1




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

    def __init__(self, definition) :
        super(StringType, self).__init__(definition)

        canopy_predicates = [dedupe.blocking.TfidfPredicate(threshold, 
                                                            self.field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

class TextType(StringType) :
    type = "Text"

    def __init__(self, definition) :
        super(TextType, self).__init__(definition)

        if 'corpus' not in definition :
            definition['corpus'] = []

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

    def __init__(self, definition) :
        super(SetType, self).__init__(definition)

        canopy_predicates = [dedupe.blocking.TfidfSetPredicate(threshold, 
                                                               self.field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

        if 'corpus' not in definition :
            definition['corpus'] = [] 

        self.comparator = dedupe.distance.CosineSetSimilarity(definition['corpus'])


class CategoricalType(FieldType) :
    type = "Categorical"
    _predicate_functions = [dedupe.predicates.wholeFieldPredicate]

    def _categories(self, definition) :
        try :
            categories = definition["categories"]
        except KeyError :
            raise ValueError('No "categories" defined')
        
        return categories

    def __init__(self, definition) :

        super(CategoricalType, self ).__init__(definition)
        
        categories = self._categories(definition)

        self.comparator = CategoricalComparator(categories)

        self.dummies = []

        for value, combo in sorted(self.comparator.combinations[2:]) :
            dummy_object = HigherDummyType({'combo' : combo, 
                                            'value' : value,
                                            'base name' : self.name,
                                            'has missing' : self.has_missing})
            self.dummies.append(dummy_object)

class SourceType(CategoricalType) :
    type = "Source"

    def _categories(self, definition) :
        try :
            categories = definition["sources"]
        except KeyError :
            raise ValueError('No "sources" defined')

        if len(categories) != 2 :
            raise ValueError("You must supply two and only " 
                             "two source names")
        
        return categories

class HigherDummyType(Variable) :
    sort_level = 1
    
    type = "HigherOrderDummy"

    def __init__(self, definition) :
        self.name = "(%s: %s)" % (str(definition['combo']), self.type)
        self.value = definition['value']
        self.base_name = definition['base name']

        super(HigherDummyType, self).__init__(definition)

class InteractionType(Variable) :
    sort_level = 2

    type = "Interaction"
    
    def __init__(self, definition) :

        self.interactions = definition["interaction variables"]

        self.name = "(Interaction: %s)" % str(self.interactions)
        self.interaction_fields = self.interactions

        super(InteractionType, self).__init__(definition)



    def expandInteractions(self, field_model) :

        self.interaction_fields = self.atomicInteractions(self.interactions,
                                                          field_model)
        for field in self.interaction_fields :
            if field_model[field].has_missing :
                self.has_missing = True

    def atomicInteractions(self, interactions, field_model) :
        atomic_interactions = []

        for field in interactions :
            try :
                field_definition = field_model[field]
            except KeyError :
                raise KeyError("The interaction variable %s is "\
                               "not a named variable in the variable "\
                               "definition" % field)
            if hasattr(field_model[field], 'interaction_fields') :
                sub_interactions = field_model[field].interaction_fields
                atomic_interactions.extend(self.atomicInteractions(sub_interactions,
                                                                   field_model))
            else :
                atomic_interactions.append(field)

        return atomic_interactions


    def dummyInteractions(self, field_model) :
        dummy_interactions = []

        categoricals = defaultdict(list)

        for field in field_model.values() :
            if hasattr(field, 'base_name') :
                if field.base_name in self.interaction_fields :
                    categoricals[field.base_name].append(field.name)

        for base_name in categoricals :
            categoricals[base_name].append(base_name)

        base_combination = set([tuple(categoricals.keys())])

        categorical_combinations = itertools.product(*categoricals.values())
        categorical_combinations = set(categorical_combinations)
        categorical_combinations -= base_combination

        non_categoricals = tuple(set(self.interaction_fields) 
                                 - set(categoricals.keys()))

        for level in categorical_combinations :
            interaction_fields = level + non_categoricals
            interaction_variable = InteractionType(
                {"interaction variables" : interaction_fields,
                 "has missing" : self.has_missing})
            interaction_variable.expandInteractions(field_model)

            dummy_interactions.append(interaction_variable)

        return dummy_interactions

class MissingDataType(Variable) :
    sort_level = 3

    type = "MissingData"

    def __init__(self, name) :
        
        self.name = "(%s: Not Missing)" %name
        self.weight = 0

        self.has_missing = False
    
class CustomType(FieldType) :
    type = "Custom"
    _predicate_functions = []

    def __init__(self, definition) :
        super(CustomType, self).__init__(definition)

        try :
            self.comparator = definition["comparator"]
        except KeyError :
            raise KeyError("For 'Custom' field types you must define "
                           "a 'comparator' function in the field "
                           "definition. ")

        if 'variable name' in definition :
            self.name = definition['variable name'] 
        else :
            self.name = "(%s: %s, %s)" % (self.field, 
                                          self.type, 
                                          self.comparator.__name__)

