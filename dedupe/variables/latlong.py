from math import sqrt

from .base import FieldType
from dedupe import predicates
from haversine import haversine


class LatLongType(FieldType):
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(x, y):
        return sqrt(haversine(x, y))
