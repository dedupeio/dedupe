from __future__ import annotations

from categorical import CategoricalComparator

from dedupe import predicates
from dedupe._typing import PredicateFunction, VariableDefinition
from dedupe.variables.base import DerivedType, FieldType


class CategoricalType(FieldType):
    type = "Categorical"
    _predicate_functions: list[PredicateFunction] = [predicates.wholeFieldPredicate]

    def _categories(self, definition: VariableDefinition) -> list[str]:
        try:
            categories = definition["categories"]
        except KeyError:
            raise ValueError('No "categories" defined')

        return categories

    def __init__(self, definition: VariableDefinition):
        super(CategoricalType, self).__init__(definition)

        categories = self._categories(definition)

        self.comparator = CategoricalComparator(categories)  # type: ignore[assignment]

        self.higher_vars = []
        for higher_var in self.comparator.dummy_names:  # type: ignore[attr-defined]
            dummy_var = DerivedType(
                {"name": higher_var, "type": "Dummy", "has missing": self.has_missing}
            )
            self.higher_vars.append(dummy_var)

    def __len__(self) -> int:
        return len(self.higher_vars)
