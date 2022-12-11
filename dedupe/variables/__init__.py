from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from dedupe.variables.base import CustomType as Custom

# flake8: noqa
from dedupe.variables.categorical_type import CategoricalType as Categorical
from dedupe.variables.date_time import DateTimeType as DateTime
from dedupe.variables.exact import ExactType as Exact
from dedupe.variables.exists import ExistsType as Exists
from dedupe.variables.interaction import InteractionType as Interaction
from dedupe.variables.latlong import LatLongType as LatLong
from dedupe.variables.price import PriceType as Price
from dedupe.variables.set import SetType as Set
from dedupe.variables.string import ShortStringType as ShortString
from dedupe.variables.string import StringType as String
from dedupe.variables.string import TextType as Text

__all__ = sorted(
    [
        "Custom",
        "Categorical",
        "DateTime",
        "Exact",
        "Exists",
        "Interaction",
        "LatLong",
        "Price",
        "Set",
        "ShortString",
        "String",
        "Text",
    ]
)
