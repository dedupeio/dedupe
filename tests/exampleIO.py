import csv
import re


def preProcess(column):
    column = re.sub("  +", " ", column)
    column = re.sub("\n", " ", column)
    column = column.strip().strip('"').strip("'").lower()
    if not column:
        column = None
    return column


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
