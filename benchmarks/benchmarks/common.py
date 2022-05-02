import csv
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

    return data_d, reader.fieldnames


def configure_logging() -> None:
    optp = optparse.OptionParser()
    optp.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="count",
        help="Increase verbosity (specify multiple times for more)",
    )
    (opts, args) = optp.parse_args()
    log_level = logging.WARNING
    if opts.verbose:
        if opts.verbose == 1:
            log_level = logging.INFO
        elif opts.verbose >= 2:
            log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)


def print_report(found_dupes, true_dupes):
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)

    print("found duplicate")
    print(len(found_dupes))

    print("precision")
    print(1 - len(false_positives) / float(len(found_dupes)))

    print("recall")
    print(len(true_positives) / float(len(true_dupes)))


def print_csv(input_file, output_file, header, clustered_dupes):
    orig_data = {}
    with open(input_file) as f:
        reader = csv.reader(f)
        reader.next()
        for row_id, row in enumerate(reader):
            orig_data[row_id] = row

    # with open("examples/output/ECP_dupes_list_" + str(time.time()) + ".csv","w") as f :
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        heading_row = header
        heading_row.insert(0, "Group_ID")
        writer.writerow(heading_row)

        dupe_id_list = []

        for group_id, cluster in enumerate(clustered_dupes, 1):
            for candidate in sorted(cluster):
                dupe_id_list.append(candidate)
                row = orig_data[candidate]
                row.insert(0, group_id)
                writer.writerow(row)

        for id in orig_data:
            if id not in set(dupe_id_list):
                row = orig_data[id]
                row.insert(0, "x")
                writer.writerow(row)
