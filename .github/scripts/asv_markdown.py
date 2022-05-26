import re


def to_markdown(data):

    preamble = """|  |       before       |    after  |       ratio | benchmark  |
|- |-: |-: |-|-|
||     {base_commit}    |    {head_commit} | | |
||     `{base_branch}`        |   `{head_branch}` | ||\n""".format(
        **data
    )

    full_table = preamble + "\n".join(
        "|" + "|".join(row) + "|" for row in data["comparisons"]
    )

    return full_table


def parse(asv_input):

    result = re.match(
        r"^       before           after         ratio\n     \[(?P<base_commit>.+)\]       \[(?P<head_commit>.+)\]\n     <(?P<base_branch>.+)>           <(?P<head_branch>.+)>\n(?P<raw_comparisons>.*)",
        asv_input,
        re.DOTALL,
    )

    test_details = result.groupdict()

    raw_comparisons = test_details.pop("raw_comparisons").splitlines()
    test_details["comparisons"] = [
        [row[:2].strip()] + row[2:].split(maxsplit=3) for row in raw_comparisons
    ]
    return test_details


if __name__ == "__main__":
    import sys

    asv_input = sys.stdin.read()
    print(asv_input, file=sys.stderr)

    print(to_markdown(parse(asv_input)))
