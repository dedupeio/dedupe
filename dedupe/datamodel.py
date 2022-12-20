from __future__ import annotations

import copyreg
import pkgutil
import types
from typing import TYPE_CHECKING, cast

import numpy

import dedupe.variables
from dedupe.variables.base import CustomType
from dedupe.variables.base import FieldType as FieldVariable
from dedupe.variables.base import MissingDataType, Variable
from dedupe.variables.interaction import InteractionType

for _, module, _ in pkgutil.iter_modules(  # type: ignore
    dedupe.variables.__path__, "dedupe.variables."
):
    __import__(module)

if TYPE_CHECKING:
    from typing import Generator, Iterable, Sequence

    from dedupe._typing import (
        Comparator,
        RecordDict,
        RecordDictPair,
        VariableDefinition,
    )
    from dedupe.predicates import Predicate

VARIABLE_CLASSES = {k: v for k, v in Variable.all_subclasses() if k}


class DataModel(object):
    version = 1

    def __init__(self, variable_definitions: Iterable[VariableDefinition]):
        variables = typify_variables(variable_definitions)
        non_interactions: list[FieldVariable] = [
            v for v in variables if not isinstance(v, InteractionType)  # type: ignore[misc]
        ]
        self.primary_variables = non_interactions
        expanded_primary = _expand_higher_variables(self.primary_variables)
        self._derived_start = len(expanded_primary)

        all_variables = expanded_primary.copy()
        all_variables += _expanded_interactions(variables)
        all_variables += missing(all_variables)

        self._missing_field_indices = missing_field_indices(all_variables)
        self._interaction_indices = interaction_indices(all_variables)

        self._len = len(all_variables)

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
        for var in self.primary_variables:
            stop = start + len(var)
            comparator = cast("Comparator", var.comparator)
            yield (var.field, comparator, start, stop)
            start = stop

    @property
    def predicates(self) -> set[Predicate]:
        predicates = set()
        for var in self.primary_variables:
            for predicate in var.predicates:
                predicates.add(predicate)
        return predicates

    def distances(
        self, record_pairs: Sequence[RecordDictPair]
    ) -> numpy.typing.NDArray[numpy.float_]:
        num_records = len(record_pairs)

        distances = numpy.empty((num_records, self._len), "f4")

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
        self, distances: numpy.typing.NDArray[numpy.float_]
    ) -> numpy.typing.NDArray[numpy.float_]:
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

        version = d.pop("version", None)
        if version is None and "_variables" in d:
            d["_len"] = len(d.pop("_variables"))
            d["primary_variables"] = d.pop("primary_fields")

        self.__dict__ = d


def typify_variables(
    variable_definitions: Iterable[VariableDefinition],
) -> list[Variable]:
    variable_definitions = list(variable_definitions)
    if not variable_definitions:
        raise ValueError("The variable definitions cannot be empty")

    variables: list[Variable] = []
    for definition in variable_definitions:
        try:
            variable_type = definition["type"]
        except TypeError:
            raise TypeError(
                "Incorrect variable specification: variable "
                "specifications are dictionaries that must "
                "include a type definition, ex. "
                "{'field' : 'Phone', type: 'String'}"
            )
        except KeyError:
            raise KeyError(
                "Missing variable type: variable "
                "specifications are dictionaries that must "
                "include a type definition, ex. "
                "{'field' : 'Phone', type: 'String'}"
            )

        if variable_type == "FuzzyCategorical" and "other fields" not in definition:
            definition["other fields"] = [  # type: ignore
                d["field"]
                for d in variable_definitions
                if ("field" in d and d["field"] != definition["field"])
            ]

        try:
            variable_class = VARIABLE_CLASSES[variable_type]
        except KeyError:
            valid = ", ".join(VARIABLE_CLASSES)
            raise KeyError(
                f"Variable type {variable_type} not valid. Valid types include {valid}"
            )
        variable_object = variable_class(definition)
        assert isinstance(variable_object, Variable)
        variables.append(variable_object)

    no_blocking_variables = all(
        isinstance(v, (CustomType, InteractionType)) for v in variables
    )
    if no_blocking_variables:
        raise ValueError(
            "At least one of the variable types needs to be a type "
            "other than 'Custom' or 'Interaction', "
            "since these types have no associated blocking rules."
        )

    return variables


def _expand_higher_variables(variables: Iterable[Variable]) -> list[Variable]:
    result: list[Variable] = []
    for variable in variables:
        if hasattr(variable, "higher_vars"):
            result.extend(variable.higher_vars)
        else:
            result.append(variable)
    return result


def missing(variables: list[Variable]) -> list[MissingDataType]:
    missing_variables = []
    for var in variables:
        if var.has_missing:
            missing_variables.append(MissingDataType(var.name))
    return missing_variables


def _expanded_interactions(variables: list[Variable]) -> list[InteractionType]:
    field_vars = {var.name: var for var in variables if isinstance(var, FieldVariable)}
    interactions = []
    for var in variables:
        if isinstance(var, InteractionType):
            var.expandInteractions(field_vars)
            interactions.extend(var.higher_vars)
    return interactions


def missing_field_indices(variables: list[Variable]) -> list[int]:
    return [i for i, var in enumerate(variables) if var.has_missing]


def interaction_indices(variables: list[Variable]) -> list[list[int]]:
    _ensure_unique_names(variables)
    name_to_index = {var.name: i for i, var in enumerate(variables)}
    indices = []
    for var in variables:
        if hasattr(var, "interaction_fields"):
            interaction_indices = [name_to_index[f] for f in var.interaction_fields]  # type: ignore
            indices.append(interaction_indices)
    return indices


def _ensure_unique_names(variables: Iterable[Variable]) -> None:
    seen = set()
    for var in variables:
        if var.name in seen:
            raise ValueError(
                "Variable name used more than once! "
                "Choose a unique name for each variable: '{var.name}'"
            )
        seen.add(var.name)


def reduce_method(m):  # type: ignore[no-untyped-def]
    return (getattr, (m.__self__, m.__func__.__name__))


copyreg.pickle(types.MethodType, reduce_method)
