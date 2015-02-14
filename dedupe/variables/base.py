from dedupe import predicates

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

class DerivedType(Variable) :
    type = "Derived"

    def __init__(self, definition) :
        self.name = "(%s: %s)" % (str(definition['name']), 
                                  str(definition['type']))
        super(DerivedType, self).__init__(definition)


class MissingDataType(Variable) :
    type = "MissingData"

    def __init__(self, name) :
        
        self.name = "(%s: Not Missing)" % name
        self.weight = 0

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

