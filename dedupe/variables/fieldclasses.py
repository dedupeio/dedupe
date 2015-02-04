import itertools
import dedupe
import dedupe.distance
import dedupe.predicates as predicates
import numpy
from collections import defaultdict

from affinegap import normalizedAffineGapDistance
from haversine import haversine
from categorical import CategoricalComparator

from dedupe.backport import OrderedDict



class Variable(object) :
    def __len__(self) :
        return 1

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
            try :
                self._predicate_functions += (predicates.existsPredicate,)
            except AttributeError :
                pass
        else :
            self.has_missing = False

class FieldType(Variable) :

    def __init__(self, definition) :
        self.field = definition['field']

        if 'variable name' in definition :
            self.name = definition['variable name'] 
        else :
            self.name = "(%s: %s)" % (self.field, self.type)

        self.predicates = [predicates.SimplePredicate(pred, self.field) 
                           for pred in self._predicate_functions]

        super(FieldType, self).__init__(definition)

class ExactType(FieldType) :
    _predicate_functions = [predicates.wholeFieldPredicate]
    type = "Exact"

    @staticmethod
    def comparator(field_1, field_2) :
        if field_1 and field_2 :
            if field_1 == field_2 :
                return 1
            else :
                return 0
        else :
            return numpy.nan

class PriceType(FieldType) :
    _predicate_functions = [predicates.orderOfMagnitude,
                            predicates.wholeFieldPredicate,
                            predicates.roundTo1]
    type = "Price"

    @staticmethod
    def comparator(price_1, price_2) :
        if price_1 <= 0 :
            return numpy.nan
        elif price_2 <= 0 :
            return numpy.nan
        else :
            return abs(numpy.log10(price_1) - numpy.log10(price_2))

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
                            dedupe.predicates.commonSixGram,
                            dedupe.predicates.commonTwoTokens,
                            dedupe.predicates.commonThreeTokens,
                            dedupe.predicates.fingerprint,
                            dedupe.predicates.oneGramFingerprint,
                            dedupe.predicates.twoGramFingerprint,
                            dedupe.predicates.sortedAcronym)

class StringType(ShortStringType) :
    comparator = normalizedAffineGapDistance
    type = "String"

    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(StringType, self).__init__(definition)

        canopy_predicates = [predicates.TfidfPredicate(threshold, 
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
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(field_1, field_2) :
        if field_1 == (0.0,0.0) or field_2 == (0.0,0.0) :
            return numpy.nan
        else :
            return haversine(field_1, field_2)

class SetType(FieldType) :
    type = "Set"

    _predicate_functions = (dedupe.predicates.wholeSetPredicate,
                            dedupe.predicates.commonSetElementPredicate,
                            dedupe.predicates.lastSetElementPredicate,
                            dedupe.predicates.commonTwoElementsPredicate,
                            dedupe.predicates.commonThreeElementsPredicate,
                            dedupe.predicates.firstSetElementPredicate)
    
    _canopy_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition) :
        super(SetType, self).__init__(definition)

        canopy_predicates = [predicates.TfidfPredicate(threshold, 
                                                       self.field)
                             for threshold in self._canopy_thresholds]

        self.predicates += canopy_predicates

        if 'corpus' not in definition :
            definition['corpus'] = [] 

        self.comparator = dedupe.distance.CosineSetSimilarity(definition['corpus'])


class CategoricalType(FieldType) :
    type = "Categorical"
    _predicate_functions = [predicates.wholeFieldPredicate]

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
  
        self.higher_vars = []
        for higher_var in self.comparator.dummy_names :
            dummy_var = DerivedType({'name' : higher_var,
                                     'type' : 'Dummy',
                                     'has missing' : self.has_missing})
            self.higher_vars.append(dummy_var)

    def __len__(self) :
        return len(self.higher_vars)


class ExistsType(CategoricalType) :
    type = "Exists"
    _predicate_functions = [predicates.existsPredicate]

    def __init__(self, definition) :

        super(CategoricalType, self ).__init__(definition)
        
        self.cat_comparator = CategoricalComparator([0,1])
  
        self.higher_vars = []
        for higher_var in self.cat_comparator.dummy_names :
            dummy_var = DerivedType({'name' : higher_var,
                                     'type' : 'Dummy',
                                     'has missing' : self.has_missing})
            self.higher_vars.append(dummy_var)

    def comparator(self, field_1, field_2) :
        if field_1 and field_2 :
            return self.cat_comparator(1, 1)
        elif field_1 or field_2 :
            return self.cat_comparator(0, 1)
        else :
            return self.cat_comparator(0, 0)

class DerivedType(Variable) :
    type = "Derived"

    def __init__(self, definition) :
        self.name = "(%s: %s)" % (str(definition['name']), 
                                  str(definition['type']))
        super(DerivedType, self).__init__(definition)


class InteractionType(Variable) :
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

        self.categorical(field_model)
    
    def categorical(self, field_model) :
        categoricals = [field for field in self.interaction_fields
                        if hasattr(field_model[field], "higher_vars")]
        noncategoricals = [field for field in self.interaction_fields
                           if not hasattr(field_model[field], "higher_vars")]

        dummies = [field_model[field].higher_vars 
                   for field in categoricals]

        self.higher_vars = []
        for combo in itertools.product(*dummies) :
            var_names = [field.name for field in combo] + noncategoricals
            higher_var = InteractionType({'has missing' : self.has_missing,
                                          'interaction variables' : var_names})
            self.higher_vars.append(higher_var)

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

class MissingDataType(Variable) :
    type = "MissingData"

    def __init__(self, name) :
        
        self.name = "(%s: Not Missing)" % name
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


def allSubclasses(cls) :
    field_classes = {}
    for q in cls.__subclasses__() :
        yield q.type, q
        for p in allSubclasses(q) :
            yield p

