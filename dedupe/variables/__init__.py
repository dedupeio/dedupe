# flake8: noqa
from dedupe.variables.base import CustomType
from dedupe.variables.categorical_type import CategoricalType
from dedupe.variables.exact import ExactType
from dedupe.variables.exists import ExistsType
from dedupe.variables.interaction import InteractionType
from dedupe.variables.latlong import LatLongType
from dedupe.variables.price import PriceType
from dedupe.variables.set import SetType
from dedupe.variables.string import ShortStringType, StringType, TextType

__all__ = [
    CustomType,
    CategoricalType,
    ExactType,
    ExistsType,
    InteractionType,
    LatLongType,
    PriceType,
    SetType,
    ShortStringType,
    StringType,
    TextType,
]
