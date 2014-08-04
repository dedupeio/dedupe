import dedupe

from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.categorical import CategoricalComparator


class FieldType(object) :
    weight = 0
    comparator = None
    _predicate_functions = []
    sort_level = 0

    def __lt__(self, other) :
        return self.sort_level < other.sort_level

             
    def __init__(self, definition) :
        self.field = definition['field']
        self.name = "%s: %s", (self.field, self.type)
        self.__hash__ = self.name

        if definition.get('Has Missing', False) :
            self.has_missing = True
        else :
            self.has_missing = False

        self.predicates = [dedupe.blocking.SimplePredicate(pred, self.field) 
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
            definition['corpus'] = None 


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

        canopy_predicates = [dedupe.blocking.TfidfSetPredicate(threshold, field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

        if 'corpus' not in definition :
            definition['corpus'] = None 

        self.comparator = dedupe.distance.CosineSetSimilarity(definition['corpus'])


class HigherDummyType(FieldType) :
    sort_level = 1

    type = "HigherOrderDummy"

    def __init__(self, definition) :
        super(HigherDummyType, self ).__init__(definition)

        self.value = definition["value"]
        
class CategoricalType(FieldType) :
    type = "Categorical"
    

    def _categories(self, definition) :
        try :
            categories = definition["Categories"]
        except KeyError :
            raise ValueError('No "Categories" defined')
        
        return categories

    def __init__(self, definition) :

        super(CategoricalType, self ).__init__(definition)
        
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
    sort_level = 2

    type = "Interaction"
    
    def __init__(self, definition, field_model) :
        super(InteractionType, self).__init__(definition)

        try :
            interactions = definition["Interaction Fields"]
        except KeyError :
            raise KeyError("""
            Missing field type: field or fields
            " "specifications are dictionaries
            that must " "name a field or fields
            to compre definition, ex. " "{'field:
            'Phone', type: 'String'}}
            """)


        self.interaction_fields = self.atomicInteractions(interactions,
                                                          field_model)
        for field in self.interaction_fields :
            if field_model[name].has_missing :
                self.has_missing = True

    def atomicInteractions(self, interactions, field_model) :
        atomic_interactions = []
        
        for field in interactions :
            if field_model[name].type == "Interaction" :
                sub_interactions = field_model[name].interaction_fields
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
            if field_model[name].type in ('Categorical', 'Source') :
                categorical_fields.add(field)
                dummies = [field]
                dummies.extend(field_model[name].higher_dummies)
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
    sort_level = 3

    type = "MissingData"

    def __init__(self, field) :
        super(MissingDataType, self).__init__({})
    

class CustomType(FieldType) :
    type = "Custom"

    def __init__(self, field, definition) :
        super(CustomType, self).__init__(definition)

        try :
            self.comparator = definition["comparator"]
        except KeyError :
            raise KeyError("For 'Custom' field types you must define "
                           "a 'comparator' function in the field "
                           "definition. ")


        self.name = "%s: %s, %s", (self.field, 
                                   self.type, 
                                   self.comparator.__name__)


