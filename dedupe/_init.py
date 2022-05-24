from dedupe.api import StaticDedupe, Dedupe  # noqa: F401
from dedupe.api import StaticRecordLink, RecordLink  # noqa: F401
from dedupe.api import StaticGazetteer, Gazetteer  # noqa: F401
from dedupe.convenience import (  # noqa: F401
    console_label,
    training_data_dedupe,
    training_data_link,
    canonicalize,
)
from dedupe.serializer import read_training, write_training  # noqa: F401
