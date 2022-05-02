from itertools import combinations
import os
import time

import dedupe

import common


def load():
    settings_file = common.DATASETS_DIR / "canonical_learned_settings"
    data_file = common.DATASETS_DIR / "restaurant-nophone-training.csv"

    data_d = common.load_data(data_file)
    training_pairs = dedupe.training_data_dedupe(data_d, "unique_id", 5000)
    true_dupes = common.get_true_dupes(data_d)

    return data_d, settings_file, training_pairs, true_dupes


def run(data: dict, settings_file, training_pairs):
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
        deduper.prepare_training(data, sample_size=10000)
        deduper.mark_pairs(training_pairs)
        deduper.train(index_predicates=True)
        with open(settings_file, "wb") as f:
            deduper.write_settings(f)

    print("clustering...")
    return deduper.partition(data, threshold=0.5)


def make_report(true_dupes, clustering):
    print("Evaluate Clustering")
    predicted_dupes = set([])
    for cluser_id, _ in clustering:
        for pair in combinations(cluser_id, 2):
            predicted_dupes.add(frozenset(pair))

    return common.Report.from_scores(true_dupes, predicted_dupes)


def cli():
    common.configure_logging()

    data, settings_file, training_pairs, true_dupes = load()

    t0 = time.time()
    clustering = run(data, settings_file, training_pairs)
    elapsed = time.time() - t0

    print(make_report(true_dupes, clustering))
    print(f"ran in {elapsed} seconds")


if __name__ == "__main__":
    cli()
