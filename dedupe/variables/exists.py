from __future__ import annotations

from typing import Callable

from categorical import CategoricalComparator

from dedupe.variables.base import DerivedType
from dedupe.variables.categorical_type import CategoricalType
from dedupe._typing import VariableDefinition


class ExistsType(CategoricalType):
    type = "Exists"
    _predicate_functions: list[Callable] = []

    def __init__(self, definition: VariableDefinition):

        super(CategoricalType, self).__init__(definition)

        self.cat_comparator = CategoricalComparator([0, 1])

        self.higher_vars = []
        for higher_var in self.cat_comparator.dummy_names:
            dummy_var = DerivedType(
                {"name": higher_var, "type": "Dummy", "has missing": self.has_missing}
            )
            self.higher_vars.append(dummy_var)

    def comparator(self, field_1, field_2) -> list[int]:
        if field_1 and field_2:
            return self.cat_comparator(1, 1)
        elif field_1 or field_2:
            return self.cat_comparator(0, 1)
        else:
            return self.cat_comparator(0, 0)

    # This flag tells fieldDistances in dedupe.core to pass
    # missing values (None) into the comparator
    comparator.missing = True  # type: ignore
