from base import FieldType
from dedupe import predicates
import numpy

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
