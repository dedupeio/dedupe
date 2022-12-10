from __future__ import annotations

from math import sqrt

from haversine import haversine

from dedupe import predicates
from dedupe.hookspecs import hookimpl
from dedupe.variables.base import FieldType


class LatLongType(FieldType):
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(x: tuple[float, float], y: tuple[float, float]) -> float:
        return sqrt(haversine(x, y))


@hookimpl
def register_variable():
    return {LatLongType.type: LatLongType}
