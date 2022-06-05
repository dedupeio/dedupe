from math import sqrt

from haversine import haversine

from dedupe import predicates
from dedupe.variables.base import FieldType


class LatLongType(FieldType):
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(x, y):
        return sqrt(haversine(x, y))
