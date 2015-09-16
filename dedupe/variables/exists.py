from .base import DerivedType
from categorical import CategoricalComparator
from .categorical_type import CategoricalType
from dedupe import predicates
import warnings

class ExistsType(CategoricalType) :
    type = "Exists"
    _predicate_functions = []

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
            if field_1 is not None :
                warnings.warn('In the dedupe 1.2 release, missing data will have to have a value of None. See http://dedupe.readthedocs.org/en/latest/Variable-definition.html#missing-data')
            return self.cat_comparator(0, 0)
            

    # This flag tells fieldDistances in dedupe.core to pass
    # missing values (None) into the comparator
    comparator.missing = True


