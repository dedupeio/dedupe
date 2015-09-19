from .base import FieldType
from dedupe import predicates
from haversine import haversine
import numpy
import warnings

class LatLongType(FieldType) :
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(field_1, field_2) :
        if field_1 == (0.0,0.0) or field_2 == (0.0,0.0) :
            warnings.warn('In the dedupe 1.2 release, missing data will have to have a value of None. See http://dedupe.readthedocs.org/en/latest/Variable-definition.html#missing-data')
            return numpy.nan
        else :
            return haversine(field_1, field_2)

