import pkgutil

import numpy
import copyreg
import types

import dedupe.variables
import dedupe.variables.base as base
from dedupe.variables.base import MissingDataType
from dedupe.variables.interaction import InteractionType

for _, module, _ in pkgutil.iter_modules(dedupe.variables.__path__,
                                         'dedupe.variables.'):
    __import__(module)


FIELD_CLASSES = {k: v for k, v in base.allSubclasses(base.FieldType) if k}


class DataModel(object):

    def __init__(self, fields):

        primary_fields, variables = typifyFields(fields)
        self.primary_fields = primary_fields
        self._derived_start = len(variables)

        variables += interactions(fields, primary_fields)
        variables += missing(variables)

        self._missing_field_indices = missing_field_indices(variables)
        self._interaction_indices = interaction_indices(variables)

        self._variables = variables

    def __len__(self):
        return len(self._variables)

    # Changing this from a property to just a normal attribute causes
    # pickling problems, because we are removing static methods from
    # their class context. This could be fixed by defining comparators
    # outside of classes in fieldclasses
    @property
    def _field_comparators(self):
        start = 0
        stop = 0
        comparators = []
        for field in self.primary_fields:
            stop = start + len(field)
            comparators.append((field.field, field.comparator, start, stop))
            start = stop

        return comparators

    def predicates(self, index_predicates=True, canopies=True):
        predicates = set()
        for definition in self.primary_fields:
            for predicate in definition.predicates:
                if hasattr(predicate, 'index'):
                    if index_predicates:
                        if hasattr(predicate, 'canopy'):
                            if canopies:
                                predicates.add(predicate)
                        else:
                            if not canopies:
                                predicates.add(predicate)
                else:
                    predicates.add(predicate)

        return predicates

    def distances(self, record_pairs):
        num_records = len(record_pairs)

        distances = numpy.empty((num_records, len(self)), 'f4')
        field_comparators = self._field_comparators

        for i, (record_1, record_2) in enumerate(record_pairs):

            for field, compare, start, stop in field_comparators:
                if record_1[field] is not None and record_2[field] is not None:
                    distances[i, start:stop] = compare(record_1[field],
                                                       record_2[field])
                elif hasattr(compare, 'missing'):
                    distances[i, start:stop] = compare(record_1[field],
                                                       record_2[field])
                else:
                    distances[i, start:stop] = numpy.nan

        distances = self._derivedDistances(distances)

        return distances

    def _derivedDistances(self, primary_distances):
        distances = primary_distances

        current_column = self._derived_start

        for interaction in self._interaction_indices:
            distances[:, current_column] =\
                numpy.prod(distances[:, interaction], axis=1)

            current_column += 1

        missing_data = numpy.isnan(distances[:, :current_column])

        distances[:, :current_column][missing_data] = 0

        if self._missing_field_indices:
            distances[:, current_column:] =\
                1 - missing_data[:, self._missing_field_indices]

        return distances

    def check(self, record):
        for field_comparator in self._field_comparators:
            field = field_comparator[0]
            if field not in record:
                raise ValueError("Records do not line up with data model. "
                                 "The field '%s' is in data_model but not "
                                 "in a record" % field)


def typifyFields(fields):
    primary_fields = []
    data_model = []

    for definition in fields:
        try:
            field_type = definition['type']
        except TypeError:
            raise TypeError("Incorrect field specification: field "
                            "specifications are dictionaries that must "
                            "include a type definition, ex. "
                            "{'field' : 'Phone', type: 'String'}")
        except KeyError:
            raise KeyError("Missing field type: fields "
                           "specifications are dictionaries that must "
                           "include a type definition, ex. "
                           "{'field' : 'Phone', type: 'String'}")

        if field_type == 'Interaction':
            continue

        if field_type == 'FuzzyCategorical' and 'other fields' not in definition:
            definition['other fields'] = [d['field'] for d in fields
                                          if ('field' in d and
                                              d['field'] != definition['field'])]

        try:
            field_class = FIELD_CLASSES[field_type]
        except KeyError:
            raise KeyError("Field type %s not valid. Valid types include %s"
                           % (definition['type'], ', '.join(FIELD_CLASSES)))

        field_object = field_class(definition)
        primary_fields.append(field_object)

        if hasattr(field_object, 'higher_vars'):
            data_model.extend(field_object.higher_vars)
        else:
            data_model.append(field_object)

    return primary_fields, data_model


def missing(data_model):
    missing_variables = []
    for definition in data_model[:]:
        if definition.has_missing:
            missing_variables.append(MissingDataType(definition.name))

    return missing_variables


def interactions(definitions, primary_fields):
    field_d = {field.name: field for field in primary_fields}
    interaction_class = InteractionType

    interactions = []

    for definition in definitions:
        if definition['type'] == 'Interaction':
            field = interaction_class(definition)
            field.expandInteractions(field_d)
            interactions.extend(field.higher_vars)

    return interactions


def missing_field_indices(variables):
    return [i for i, definition
            in enumerate(variables)
            if definition.has_missing]


def interaction_indices(variables):
    indices = []

    field_names = [field.name for field in variables]

    for definition in variables:
        if hasattr(definition, 'interaction_fields'):
            interaction_indices = []
            for interaction_field in definition.interaction_fields:
                interaction_indices.append(
                    field_names.index(interaction_field))
            indices.append(interaction_indices)

    return indices


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


copyreg.pickle(types.MethodType, reduce_method)
