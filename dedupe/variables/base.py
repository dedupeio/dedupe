from __future__ import annotations

from typing import TYPE_CHECKING

from dedupe import predicates

if TYPE_CHECKING:
    from typing import Any, ClassVar, Iterable, Sequence

    from dedupe._typing import Comparator, CustomComparator, PredicateFunction
    from dedupe._typing import Variable as VariableProtocol


class Variable:
    name: str
    type: ClassVar[str]
    predicates: list[predicates.Predicate]
    higher_vars: Sequence[VariableProtocol]

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        other_name: str = other.name
        return self.name == other_name

    def __init__(self, has_missing: bool = False):
        self.has_missing = has_missing

    def __getstate__(self) -> dict[str, Any]:
        odict = self.__dict__.copy()
        odict["predicates"] = None

        return odict


class DerivedType(Variable):
    type = "Derived"

    def __init__(self, name: str, var_type: str, **kwargs):
        self.name = "({}: {})".format(str(name), str(var_type))
        super().__init__(**kwargs)


class FieldType(Variable):
    _index_thresholds: Sequence[float] = []
    _index_predicates: Sequence[type[predicates.IndexPredicate]] = []
    _predicate_functions: Sequence[PredicateFunction] = ()
    _Predicate: type[predicates.SimplePredicate] = predicates.SimplePredicate
    comparator: Comparator

    def __init__(self, field: str, name: str | None = None, has_missing: bool = False):
        self.field = field

        if name is None:
            self.name = "({}: {})".format(self.field, self.type)
        else:
            self.name = name

        self.predicates = [
            self._Predicate(pred, self.field) for pred in self._predicate_functions
        ]

        self.predicates += indexPredicates(
            self._index_predicates, self._index_thresholds, self.field
        )

        self.has_missing = has_missing
        if self.has_missing:
            exists_pred = predicates.ExistsPredicate(self.field)
            self.predicates.append(exists_pred)


class CustomType(FieldType):
    type = "Custom"

    def __init__(
        self,
        field: str,
        comparator: CustomComparator,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(field, **kwargs)

        if comparator is None:
            raise ValueError(
                "You must define a comparator function for the Custom class"
            )
        else:
            self.comparator = comparator

        if name is None:
            self.name = "({}: {}, {})".format(
                self.field,
                self.type,
                self.comparator.__name__,
            )
        else:
            self.name = name


def indexPredicates(
    predicates: Iterable[type[predicates.IndexPredicate]],
    thresholds: Sequence[float],
    field: str,
) -> list[predicates.IndexPredicate]:
    index_predicates = []
    for predicate in predicates:
        for threshold in thresholds:
            index_predicates.append(predicate(threshold, field))

    return index_predicates
