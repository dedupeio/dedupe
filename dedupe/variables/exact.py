from .base import FieldType
from dedupe import predicates
import numpy
import warnings

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
            warnings.warn('In the dedupe 1.2 release, missing data will have to have a value of None. See http://dedupe.readthedocs.org/en/latest/Variable-definition.html#missing-data')
            return numpy.nan
