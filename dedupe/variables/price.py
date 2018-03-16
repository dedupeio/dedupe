import numpy
from dedupe import predicates
from .base import FieldType


class PriceType(FieldType):
    _predicate_functions = [predicates.orderOfMagnitude,
                            predicates.wholeFieldPredicate,
                            predicates.roundTo1]
    type = "Price"

    @staticmethod
    def comparator(price_1, price_2):
        if price_1 <= 0:
            return numpy.nan
        elif price_2 <= 0:
            return numpy.nan
        else:
            return abs(numpy.log10(price_1) - numpy.log10(price_2))
