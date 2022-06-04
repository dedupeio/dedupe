import pkgutil
from typing import Container, Iterable

import numpy
import copyreg
import types

import dedupe.variables
from dedupe.variables.base import (
    FieldType as FieldVariable,
    MissingDataType,
    Variable,
)
from dedupe.variables.interaction import InteractionType
from dedupe._typing import VariableDefinition

for _, module, _ in pkgutil.iter_modules(  # type: ignore
    dedupe.variables.__path__, "dedupe.variables."
):
    __import__(module)

VARIABLE_CLASSES = {k: v for k, v in FieldVariable.all_subclasses() if k}


class DataModel(object):
    def __init__(self, variable_definitions: Iterable[VariableDefinition]):
        variable_definitions = list(variable_definitions)
        if not variable_definitions:
            raise ValueError("The variable definitions cannot be empty")
        all_variables: list[Variable]
        self.primary_variables, all_variables = typify_variables(variable_definitions)
        self._derived_start = len(all_variables)

        all_variables += interactions(variable_definitions, self.primary_variables)
        all_variables += missing(all_variables)

        self._missing_field_indices = missing_field_indices(all_variables)
        self._interaction_indices = interaction_indices(all_variables)

        self._len = len(all_variables)

    def __len__(self):
        return self._len

    # Changing this from a property to just a normal attribute causes
    # pickling problems, because we are removing static methods from
    # their class context. This could be fixed by defining comparators
    # outside of classes in fieldclasses
    @property
    def _field_comparators(self):
        start = 0
        stop = 0
        for var in self.primary_variables:
            stop = start + len(var)
            yield (var.field, var.comparator, start, stop)
            start = stop

    def predicates(self, index_predicates=True, canopies=True) -> set:
        predicates = set()
        for var in self.primary_variables:
            for predicate in var.predicates:
                if hasattr(predicate, "index"):
                    if index_predicates:
                        if hasattr(predicate, "canopy"):
                            if canopies:
                                predicates.add(predicate)
                        else:
                            if not canopies:
                                predicates.add(predicate)
                else:
                    predicates.add(predicate)

        return predicates

    def distances(self, record_pairs) -> numpy.ndarray:
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

    def _add_derived_distances(self, distances: numpy.ndarray) -> numpy.ndarray:
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

    def check(self, record: Container) -> None:
        for field, _, _, _ in self._field_comparators:
            if field not in record:
                raise ValueError(
                    "Records do not line up with data model. "
                    "The field '%s' is in data_model but not "
                    "in a record" % field
                )


def typify_variables(
    variable_definitions: Iterable[VariableDefinition],
) -> tuple[list[FieldVariable], list[Variable]]:
    primary_variables = []
    all_variables = []
    only_custom = True

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

        if variable_type != "Custom":
            only_custom = False

        if variable_type == "Interaction":
            continue

        if variable_type == "FuzzyCategorical" and "other fields" not in definition:
            definition["other fields"] = [  # type: ignore
                d["field"]
                for d in variable_definitions
                if ("field" in d and d["field"] != definition["field"])
            ]

        try:
            variable_class = VARIABLE_CLASSES[variable_type]
        except KeyError:
            raise KeyError(
                "Field type %s not valid. Valid types include %s"
                % (definition["type"], ", ".join(VARIABLE_CLASSES))
            )

        variable_object = variable_class(definition)
        primary_variables.append(variable_object)

        if hasattr(variable_object, "higher_vars"):
            all_variables.extend(variable_object.higher_vars)  # type: ignore
        else:
            all_variables.append(variable_object)

    if only_custom:
        raise ValueError(
            "At least one of the variable types needs to be a type"
            "other than 'Custom'. 'Custom' types have no associated"
            "blocking rules"
        )

    return primary_variables, all_variables


def missing(variables: list[Variable]) -> list[MissingDataType]:
    missing_variables = []
    for var in variables:
        if var.has_missing:
            missing_variables.append(MissingDataType(var.name))
    return missing_variables


def interactions(
    definitions: Iterable[VariableDefinition], primary_variables: list[FieldVariable]
) -> list[InteractionType]:
    field_d = {field.name: field for field in primary_variables}

    interactions = []
    for definition in definitions:
        if definition["type"] == "Interaction":
            var = InteractionType(definition)
            var.expandInteractions(field_d)
            interactions.extend(var.higher_vars)
    return interactions


def missing_field_indices(variables: list[Variable]) -> list[int]:
    return [i for i, var in enumerate(variables) if var.has_missing]


def interaction_indices(variables: list[Variable]) -> list[list[int]]:
    var_names = [var.name for var in variables]
    indices = []
    for var in variables:
        if hasattr(var, "interaction_fields"):
            interaction_indices = [var_names.index(f) for f in var.interaction_fields]  # type: ignore
            indices.append(interaction_indices)
    return indices


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


copyreg.pickle(types.MethodType, reduce_method)
