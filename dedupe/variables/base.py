from __future__ import annotations

from typing import TYPE_CHECKING

from dedupe import predicates

if TYPE_CHECKING:
    from typing import Any, ClassVar, Generator, Iterable, Optional, Sequence, Type

    from dedupe._typing import Comparator, PredicateFunction, VariableDefinition


class Variable(object):
    name: str
    type: ClassVar[str]
    predicates: list[predicates.Predicate]
    higher_vars: Sequence["Variable"]

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        other_name: str = other.name
        return self.name == other_name

    def __init__(self, definition: VariableDefinition):
        if definition.get("has missing", False):
            self.has_missing = True
            try:
                exists_pred = predicates.ExistsPredicate(definition["field"])
                self.predicates.append(exists_pred)
            except KeyError:
                pass
        else:
            self.has_missing = False

    def __getstate__(self) -> dict[str, Any]:
        odict = self.__dict__.copy()
        odict["predicates"] = None

        return odict

    @classmethod
    def all_subclasses(
        cls,
    ) -> Generator[tuple[Optional[str], Type["Variable"]], None, None]:
        for q in cls.__subclasses__():
            yield getattr(q, "type", None), q
            for p in q.all_subclasses():
                yield p


class DerivedType(Variable):
    type = "Derived"

    def __init__(self, definition: VariableDefinition):
        self.name = "(%s: %s)" % (str(definition["name"]), str(definition["type"]))
        super(DerivedType, self).__init__(definition)


class MissingDataType(Variable):
    type = "MissingData"

    def __init__(self, name: str):
        self.name = "(%s: Not Missing)" % name

        self.has_missing = False


class FieldType(Variable):
    _index_thresholds: Sequence[float] = []
    _index_predicates: Sequence[Type[predicates.IndexPredicate]] = []
    _predicate_functions: Sequence[PredicateFunction] = ()
    _Predicate: Type[predicates.SimplePredicate] = predicates.SimplePredicate
    comparator: Comparator

    def __init__(self, definition: VariableDefinition):
        self.field = definition["field"]

        if "variable name" in definition:
            self.name = definition["variable name"]
        else:
            self.name = "(%s: %s)" % (self.field, self.type)

        self.predicates = [
            self._Predicate(pred, self.field) for pred in self._predicate_functions
        ]

        self.predicates += indexPredicates(
            self._index_predicates, self._index_thresholds, self.field
        )

        super(FieldType, self).__init__(definition)


class CustomType(FieldType):
    type = "Custom"

    def __init__(self, definition: VariableDefinition):
        super(CustomType, self).__init__(definition)

        try:
            self.comparator = definition["comparator"]  # type: ignore[assignment]
        except KeyError:
            raise KeyError(
                "For 'Custom' field types you must define "
                "a 'comparator' function in the field "
                "definition. "
            )

        if "variable name" not in definition:
            self.name = "(%s: %s, %s)" % (
                self.field,
                self.type,
                self.comparator.__name__,
            )


def indexPredicates(
    predicates: Iterable[Type[predicates.IndexPredicate]],
    thresholds: Sequence[float],
    field: str,
) -> list[predicates.IndexPredicate]:
    index_predicates = []
    for predicate in predicates:
        for threshold in thresholds:
            index_predicates.append(predicate(threshold, field))

    return index_predicates
