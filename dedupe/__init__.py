from dedupe.api import (  # noqa: F401
    Dedupe,
    Gazetteer,
    RecordLink,
    StaticDedupe,
    StaticGazetteer,
    StaticRecordLink,
)
from dedupe.convenience import (  # noqa: F401
    canonicalize,
    console_label,
    training_data_dedupe,
    training_data_link,
)
from dedupe.serializer import read_training, write_training  # noqa: F401

__all__ = [
    "Dedupe",
    "Gazetteer",
    "RecordLink",
    "StaticDedupe",
    "StaticGazetteer",
    "StaticRecordLink",
    "canonicalize",
    "console_label",
    "training_data_dedupe",
    "training_data_link",
    "read_training",
    "write_training",
]
