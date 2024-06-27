from __future__ import annotations

import itertools
from typing import Mapping

from dedupe._typing import FieldVariable, InteractionVariable
from dedupe.variables.base import Variable


class InteractionType(Variable):
    type = "Interaction"
    higher_vars: list[InteractionVariable]

    def __init__(self, *args: str, **kwargs):
        self.interactions = list(args)

        self.name = "(Interaction: %s)" % str(self.interactions)
        self.interaction_fields = self.interactions

        super().__init__(**kwargs)

    def expandInteractions(self, field_model: Mapping[str, FieldVariable]) -> None:
        self.interaction_fields = self.atomicInteractions(
            self.interactions, field_model
        )
        for field in self.interaction_fields:
            if field_model[field].has_missing:
                self.has_missing = True

        self.categorical(field_model)

    def categorical(self, field_model: Mapping[str, FieldVariable]) -> None:
        categoricals = [
            field
            for field in self.interaction_fields
            if hasattr(field_model[field], "higher_vars")
        ]
        noncategoricals = [
            field
            for field in self.interaction_fields
            if not hasattr(field_model[field], "higher_vars")
        ]

        dummies = [field_model[field].higher_vars for field in categoricals]  # type: ignore[attr-defined]

        self.higher_vars = []
        for combo in itertools.product(*dummies):
            var_names = [field.name for field in combo] + noncategoricals
            higher_var = InteractionType(*var_names, has_missing=self.has_missing)
            self.higher_vars.append(higher_var)

    def atomicInteractions(
        self, interactions: list[str], field_model: Mapping[str, FieldVariable]
    ) -> list[str]:
        atomic_interactions = []

        for field in interactions:
            try:
                field_model[field]
            except KeyError:
                raise KeyError(
                    "The interaction variable %s is "
                    "not a named variable in the variable "
                    "definition" % field
                )

            if hasattr(field_model[field], "interaction_fields"):
                sub_interactions = field_model[field].interaction_fields  # type: ignore[attr-defined]
                atoms = self.atomicInteractions(sub_interactions, field_model)
                atomic_interactions.extend(atoms)
            else:
                atomic_interactions.append(field)

        return atomic_interactions
