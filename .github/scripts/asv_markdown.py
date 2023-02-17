import re


def format(element):
    if is_float(element):
        f = float(element)

        return "{0:.3}".format(f)

    else:
        return element


def is_float(element):
    try:
        float(element)
    except ValueError:
        return False
    else:
        return True


def to_markdown(data):
    preamble = """# {tests} ([diff](https://github.com/dedupeio/dedupe/compare/{base_commit}...{head_commit})):
|  |       before       |    after  |       ratio | benchmark  |
|- |-: |-: |-: |-|\n""".format(
        **data
    )

    full_table = preamble + "\n".join(
        "|" + "|".join(row) + "|" for row in data["comparisons"]
    )

    return full_table


def parse(asv_input):
    result = re.match(
        r"^\n(?P<tests>.*?):\n\n       before           after         ratio\n     \[(?P<base_commit>.+)\]       \[(?P<head_commit>.+)\]\n     <(?P<base_branch>.+)>           <(?P<head_branch>.+)> *\n(?P<raw_comparisons>.*)",
        asv_input,
        re.DOTALL,
    )

    test_details = result.groupdict()

    raw_comparisons = test_details.pop("raw_comparisons").splitlines()
    comparisons = (
        [row[:2].strip()] + row[2:].split(maxsplit=3) for row in raw_comparisons
    )
    test_details["comparisons"] = [
        [indicator, format(value_a), format(value_b), ratio, test]
        for indicator, value_a, value_b, ratio, test in comparisons
    ]
    return test_details


if __name__ == "__main__":
    import sys

    print("hello", file=sys.stderr)
    asv_input = sys.stdin.read()
    print(asv_input, file=sys.stderr)

    print(to_markdown(parse(asv_input)))
