from .base import FieldType, DerivedType
from dedupe import predicates
from categorical import CategoricalComparator

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

