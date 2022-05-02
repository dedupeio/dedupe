from itertools import combinations, groupby
import os
import time

import dedupe

import common


if __name__ == "__main__":
    common.configure_logging()

    settings_file = common.DATASETS_DIR / "canonical_learned_settings"
    raw_data = common.DATASETS_DIR / "restaurant-nophone-training.csv"

    data_d, header = common.load_data(raw_data)

    training_pairs = dedupe.training_data_dedupe(data_d, "unique_id", 5000)

    duplicates = set()
    for _, pair in groupby(
        sorted(data_d.items(), key=lambda x: x[1]["unique_id"]),
        key=lambda x: x[1]["unique_id"],
    ):
        pair = list(pair)
        if len(pair) == 2:
            a, b = pair
            duplicates.add(frozenset((a[0], b[0])))

    t0 = time.time()

    print("number of known duplicate pairs", len(duplicates))

    if os.path.exists(settings_file):
        with open(settings_file, "rb") as f:
            deduper = dedupe.StaticDedupe(f)

    else:
        fields = [
            {"field": "name", "type": "String"},
            {"field": "name", "type": "Exact"},
            {"field": "address", "type": "String"},
            {"field": "cuisine", "type": "ShortString", "has missing": True},
            {"field": "city", "type": "ShortString"},
        ]

        deduper = dedupe.Dedupe(fields, num_cores=5)
        deduper.prepare_training(data_d, sample_size=10000)
        deduper.mark_pairs(training_pairs)
        deduper.train(index_predicates=True)
        with open(settings_file, "wb") as f:
            deduper.write_settings(f)

    # print candidates
    print("clustering...")
    clustered_dupes = deduper.partition(data_d, threshold=0.5)

    print("Evaluate Clustering")
    confirm_dupes = set([])
    for dupes, score in clustered_dupes:
        for pair in combinations(dupes, 2):
            confirm_dupes.add(frozenset(pair))

    print(common.Report.from_scores(duplicates, confirm_dupes))

    print("ran in ", time.time() - t0, "seconds")
