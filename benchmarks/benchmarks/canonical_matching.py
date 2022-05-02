import itertools
import os
import time

import dedupe

import common


if __name__ == "__main__":
    common.configure_logging()

    settings_file = common.DATASETS_DIR / "canonical_data_matching_learned_settings"

    data_1, header = common.load_data(common.DATASETS_DIR / "restaurant-1.csv")
    data_2, _ = common.load_data(common.DATASETS_DIR / "restaurant-2.csv")

    training_pairs = dedupe.training_data_link(data_1, data_2, "unique_id", 5000)

    all_data = data_1.copy()
    all_data.update(data_2)

    duplicates_s = set()
    for _, pair in itertools.groupby(
        sorted(all_data.items(), key=lambda x: x[1]["unique_id"]),
        key=lambda x: x[1]["unique_id"],
    ):
        pair = list(pair)
        if len(pair) == 2:
            a, b = pair
            duplicates_s.add(frozenset((a[0], b[0])))

    t0 = time.time()

    print("number of known duplicate pairs", len(duplicates_s))

    if os.path.exists(settings_file):
        with open(settings_file, "rb") as f:
            deduper = dedupe.StaticRecordLink(f)
    else:
        fields = [
            {"field": "name", "type": "String"},
            {"field": "address", "type": "String"},
            {"field": "cuisine", "type": "String"},
            {"field": "city", "type": "String"},
        ]

        deduper = dedupe.RecordLink(fields)
        deduper.prepare_training(data_1, data_2, sample_size=10000)
        deduper.mark_pairs(training_pairs)
        deduper.train()

        with open(settings_file, "wb") as f:
            deduper.write_settings(f)

    # print candidates
    print("clustering...")
    clustered_dupes = deduper.join(data_1, data_2, threshold=0.5)

    print("Evaluate Clustering")
    confirm_dupes = set(frozenset(pair) for pair, score in clustered_dupes)

    common.print_report(confirm_dupes, duplicates_s)

    print("ran in ", time.time() - t0, "seconds")

    # print candidates
    print("clustering...")
    clustered_dupes = deduper.join(
        data_1, data_2, threshold=0.5, constraint="many-to-one"
    )

    print("Evaluate Clustering")
    confirm_dupes = set(frozenset(pair) for pair, score in clustered_dupes)

    common.print_report(confirm_dupes, duplicates_s)

    print("ran in ", time.time() - t0, "seconds")
