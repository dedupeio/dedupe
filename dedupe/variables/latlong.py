from base import FieldType
from dedupe import predicates
from haversine import haversine
import numpy

class LatLongType(FieldType) :
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(field_1, field_2) :
        if field_1 == (0.0,0.0) or field_2 == (0.0,0.0) :
            return numpy.nan
        else :
            return haversine(field_1, field_2)

