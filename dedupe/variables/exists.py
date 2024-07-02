from __future__ import annotations

from typing import Any

from categorical import CategoricalComparator

from dedupe._typing import PredicateFunction
from dedupe.variables.base import DerivedType, FieldType


class ExistsType(FieldType):
    type = "Exists"
    _predicate_functions: list[PredicateFunction] = []

    def __init__(self, field: str, **kwargs):
        super().__init__(field, **kwargs)

        self.cat_comparator = CategoricalComparator([0, 1])

        self.higher_vars = []
        for higher_var in self.cat_comparator.dummy_names:
            dummy_var = DerivedType(higher_var, "Dummy", has_missing=self.has_missing)
            self.higher_vars.append(dummy_var)

    def comparator(self, field_1: Any, field_2: Any) -> list[int]:
        if field_1 and field_2:
            return self.cat_comparator(1, 1)
        elif field_1 or field_2:
            return self.cat_comparator(0, 1)
        else:
            return self.cat_comparator(0, 0)

    def __len__(self) -> int:
        return len(self.higher_vars)

    # This flag tells fieldDistances in dedupe.core to pass
    # missing values (None) into the comparator
    comparator.missing = True  # type: ignore
