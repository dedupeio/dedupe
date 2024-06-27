from __future__ import annotations

from typing import Sequence

from categorical import CategoricalComparator

from dedupe import predicates
from dedupe._typing import PredicateFunction
from dedupe.variables.base import DerivedType, FieldType


class CategoricalType(FieldType):
    type = "Categorical"
    _predicate_functions: list[PredicateFunction] = [predicates.wholeFieldPredicate]

    def __init__(self, field: str, categories: Sequence[str], **kwargs):
        super().__init__(field, **kwargs)

        self.comparator = CategoricalComparator(categories)  # type: ignore[assignment]

        self.higher_vars = []
        for higher_var in self.comparator.dummy_names:  # type: ignore[attr-defined]
            dummy_var = DerivedType(higher_var, "Dummy", has_missing=False)
            self.higher_vars.append(dummy_var)

    def __len__(self) -> int:
        return len(self.higher_vars)
