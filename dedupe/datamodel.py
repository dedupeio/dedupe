from __future__ import annotations

import copyreg
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy

from dedupe._typing import FieldVariable
from dedupe.variables.interaction import InteractionType

if TYPE_CHECKING:
    from typing import Collection, Generator, Iterable, Sequence

    from dedupe._typing import (
        Comparator,
        InteractionVariable,
        RecordDict,
        RecordDictPair,
        Variable,
    )
    from dedupe.predicates import Predicate


class DataModel:
    version = 2

    def __init__(self, variable_definitions: Collection[Variable]):
        for item in variable_definitions:
            if isinstance(item, Mapping):
                raise ValueError(
                    "It looks like you are trying to use a variable definition "
                    "composed of dictionaries. dedupe 3.0 uses variable objects "
                    'directly. So instead of [{"field": "name", "type": "String"}] '
                    'we now do [dedupe.variables.String("name")].'
                )

        variable_definitions = list(variable_definitions)
        if not variable_definitions:
            raise ValueError("The variable definitions cannot be empty")
        if not any(variable.predicates for variable in variable_definitions):
            raise ValueError(
                "At least one of the variable types needs to be a type"
                "other than 'Custom'. 'Custom' types have no associated"
                "blocking rules"
            )

        # This is a protocol check, not a class inheritance check
        self.field_variables: list[FieldVariable] = [
            variable
            for variable in variable_definitions
            if isinstance(variable, FieldVariable)
        ]

        # we need to keep track of ordering of variables because in
        # order to calculate derived fields like interaction and missing
        # data fields.
        columns: list[Variable] = []
        for variable in self.field_variables:
            if len(variable) == 1:
                columns.append(variable)
            elif len(variable) > 1:
                assert hasattr(variable, "higher_vars")
                columns.extend(variable.higher_vars)

        self._derived_start = len(columns)

        # i'm not really satisfied with how we are dealing with interactions
        # here. seems like there should be a cleaner path, but i don't see it
        # today
        columns += interactions(variable_definitions, self.field_variables)

        self._missing_field_indices = missing_field_indices(columns)
        self._interaction_indices = interaction_indices(columns)

        self._len = len(columns) + len(self._missing_field_indices)

    def __len__(self) -> int:
        return self._len

    # Changing this from a property to just a normal attribute causes
    # pickling problems, because we are removing static methods from
    # their class context. This could be fixed by defining comparators
    # outside of classes in fieldclasses
    @property
    def _field_comparators(
        self,
    ) -> Generator[tuple[str, Comparator, int, int], None, None]:
        start = 0
        stop = 0
        for var in self.field_variables:
            stop = start + len(var)
            comparator = cast("Comparator", var.comparator)
            yield (var.field, comparator, start, stop)
            start = stop

    @property
    def predicates(self) -> set[Predicate]:
        predicates = set()
        for var in self.field_variables:
            for predicate in var.predicates:
                predicates.add(predicate)
        return predicates

    def distances(
        self, record_pairs: Sequence[RecordDictPair]
    ) -> numpy.typing.NDArray[numpy.float64]:
        num_records = len(record_pairs)

        distances = numpy.empty((num_records, len(self)), "f4")

        for i, (record_1, record_2) in enumerate(record_pairs):
            for field, compare, start, stop in self._field_comparators:
                if record_1[field] is not None and record_2[field] is not None:
                    distances[i, start:stop] = compare(record_1[field], record_2[field])
                elif hasattr(compare, "missing"):
                    distances[i, start:stop] = compare(record_1[field], record_2[field])
                else:
                    distances[i, start:stop] = numpy.nan

        distances = self._add_derived_distances(distances)

        return distances

    def _add_derived_distances(
        self, distances: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        current_column = self._derived_start

        for indices in self._interaction_indices:
            distances[:, current_column] = numpy.prod(distances[:, indices], axis=1)
            current_column += 1

        is_missing = numpy.isnan(distances[:, :current_column])

        distances[:, :current_column][is_missing] = 0

        if self._missing_field_indices:
            distances[:, current_column:] = (
                1 - is_missing[:, self._missing_field_indices]
            )

        return distances

    def check(self, record: RecordDict) -> None:
        for field, _, _, _ in self._field_comparators:
            if field not in record:
                raise ValueError(
                    "Records do not line up with data model. "
                    "The field '%s' is in data_model but not "
                    "in a record" % field
                )

    def __getstate__(self):
        d = self.__dict__
        d["object_version"] = self.version
        return d

    def __setstate__(self, d):
        version = d.pop("object_version", None)
        if version is None and "_variables" in d:
            d["_len"] = len(d.pop("_variables"))
            d["primary_variables"] = d.pop("primary_fields")
        elif version == 1:
            d["field_variables"] = d.pop("primary_variables")

        self.__dict__ = d


def interactions(
    variables: Iterable[Variable], primary_variables: Iterable[FieldVariable]
) -> list[InteractionVariable]:
    field_d = {field.name: field for field in primary_variables}

    interactions: list[InteractionVariable] = []
    for variable in variables:
        if isinstance(variable, InteractionType):
            variable.expandInteractions(field_d)
            interactions.extend(variable.higher_vars)
    return interactions


def missing_field_indices(variables: list[Variable]) -> list[int]:
    return [i for i, var in enumerate(variables) if var.has_missing]


def interaction_indices(variables: list[Variable]) -> list[list[int]]:
    var_names = [var.name for var in variables]
    indices = []
    for var in variables:
        if hasattr(var, "interaction_fields"):
            interaction_indices = [var_names.index(f) for f in var.interaction_fields]
            indices.append(interaction_indices)
    return indices


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


copyreg.pickle(types.MethodType, reduce_method)
