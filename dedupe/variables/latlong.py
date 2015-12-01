from .base import FieldType
from dedupe import predicates
from haversine import haversine

class LatLongType(FieldType) :
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    comparator = haversine

