import csv
from dataclasses import dataclass
from itertools import groupby
import logging
import optparse
from pathlib import Path
import re


DATASETS_DIR = Path(__file__).parent / "datasets"


def pre_process(column):
    column = re.sub("  +", " ", column)
    column = re.sub("\n", " ", column)
    column = column.strip().strip('"').strip("'").lower()
    if not column:
        column = None
    return column


def load_data(pathlike):
    data_d = {}
    with open(pathlike) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = {k: pre_process(v) for (k, v) in row.items()}
            data_d[str(pathlike) + str(i)] = clean_row

    return data_d


def configure_logging() -> None:
    optp = optparse.OptionParser()
    optp.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="count",
        help="Increase verbosity (specify multiple times for more)",
    )
    opts, _ = optp.parse_args()
    log_level = logging.WARNING
    if opts.verbose:
        if opts.verbose == 1:
            log_level = logging.INFO
        elif opts.verbose >= 2:
            log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)


def get_true_dupes(data: dict) -> set:
    duplicates = set()
    for _, pair in groupby(
        sorted(data.items(), key=lambda x: x[1]["unique_id"]),
        key=lambda x: x[1]["unique_id"],
    ):
        pair = list(pair)
        if len(pair) == 2:
            a, b = pair
            duplicates.add(frozenset((a[0], b[0])))
    return duplicates


@dataclass
class Report:
    # TODO add more and replace calculations with sklearn
    n_true: int
    n_found: int
    precision: float
    recall: float

    @classmethod
    def from_scores(cls, true_dupes: set, found_dupes: set):
        true_positives = found_dupes.intersection(true_dupes)

        n_true = len(true_dupes)
        n_found = len(found_dupes)
        precision = len(true_positives) / n_found
        recall = len(true_positives) / n_true

        return cls(n_true, n_found, precision, recall)
