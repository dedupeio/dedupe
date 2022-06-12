from __future__ import annotations

import numpy as np

from dedupe import predicates
from dedupe.variables.base import FieldType


class PriceType(FieldType):
    _predicate_functions = [
        predicates.orderOfMagnitude,
        predicates.wholeFieldPredicate,
        predicates.roundTo1,
    ]
    type = "Price"

    @staticmethod
    def comparator(price_1: int | float, price_2: int | float) -> float:
        if price_1 <= 0:
            return np.nan
        elif price_2 <= 0:
            return np.nan
        else:
            return abs(np.log10(price_1) - np.log10(price_2))
