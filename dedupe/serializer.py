import json
from typing import Any, Iterator, TextIO

from dedupe._typing import TrainingData


def _from_json(json_object: Any) -> Any:
    if "__class__" in json_object:
        if json_object["__class__"] == "frozenset":
            return frozenset(json_object["__value__"])
        if json_object["__class__"] == "tuple":
            return tuple(json_object["__value__"])
    return json_object


def hint_tuples(item: Any) -> Any:
    if isinstance(item, tuple):
        return {"__class__": "tuple", "__value__": [hint_tuples(e) for e in item]}
    if isinstance(item, list):
        return [hint_tuples(e) for e in item]
    if isinstance(item, dict):
        return {key: hint_tuples(value) for key, value in item.items()}
    else:
        return item


class TupleEncoder(json.JSONEncoder):
    def encode(self, obj: Any) -> Any:
        return super().encode(hint_tuples(obj))

    def iterencode(self, obj: Any, _one_shot: bool = False) -> Iterator[str]:
        return super().iterencode(hint_tuples(obj))

    def default(self, python_object: Any) -> Any:
        if isinstance(python_object, frozenset):
            return {"__class__": "frozenset", "__value__": list(python_object)}
        return super().default(python_object)


def read_training(training_file: TextIO) -> Any:
    """
    Read training from previously built training data file object

    Args:
        training_file: file object containing the training data

    Returns:
        A dictionary with two keys, `match` and `distinct`. See the inverse,
        :func:`write_training`.
    """
    return json.load(training_file, object_hook=_from_json)


def write_training(labeled_pairs: TrainingData, file_obj: TextIO) -> None:
    """
    Write a JSON file that contains labeled examples

    Args:
        labeled_pairs: A dictionary with two keys, `match` and `distinct`.
                       The values are lists that can contain pairs of records
        file_obj: file object to write training data to

    .. code:: python

        examples = {
            "match": [
                 ({'name' : 'Georgie Porgie'}, {'name' : 'George Porgie'}),
            ],
            "distinct": [
                ({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'}),
            ],
        }
        with open('training.json', 'w') as f:
            dedupe.write_training(examples, f)

    """
    json.dump(labeled_pairs, file_obj, cls=TupleEncoder, ensure_ascii=True)
