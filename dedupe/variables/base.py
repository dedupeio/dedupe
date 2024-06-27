from __future__ import annotations

from typing import TYPE_CHECKING

from dedupe import predicates

if TYPE_CHECKING:
    from typing import Any, ClassVar, Generator, Iterable, Optional, Sequence, Type

    from dedupe._typing import Comparator, PredicateFunction
    from dedupe._typing import Variable as VariableProtocol


class Variable(object):
    name: str
    type: ClassVar[str]
    predicates: list[predicates.Predicate]
    higher_vars: Sequence["VariableProtocol"]

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        other_name: str = other.name
        return self.name == other_name

    def __init__(self, has_missing=False):
        self.has_missing = has_missing

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

    def __init__(self, name, var_type, **kwargs):
        self.name = "(%s: %s)" % (str(name), str(var_type))
        super().__init__(**kwargs)


class FieldType(Variable):
    _index_thresholds: Sequence[float] = []
    _index_predicates: Sequence[Type[predicates.IndexPredicate]] = []
    _predicate_functions: Sequence[PredicateFunction] = ()
    _Predicate: Type[predicates.SimplePredicate] = predicates.SimplePredicate
    comparator: Comparator

    def __init__(self, field, name=None, has_missing=False):
        self.field = field

        if name is None:
            self.name = "(%s: %s)" % (self.field, self.type)
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

    def __init__(self, field, comparator, name=None, **kwargs):
        super().__init__(field, **kwargs)

        if comparator is None:
            raise ValueError(
                "You must define a comparator function for the Custom class"
            )
        else:
            self.comparator = comparator

        if name is None:
            self.name = "(%s: %s, %s)" % (
                self.field,
                self.type,
                self.comparator.__name__,
            )
        else:
            self.name = name


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
