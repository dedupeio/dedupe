import itertools
import os
import time

import dedupe

import common


def load():
    settings_file = common.DATASETS_DIR / "canonical_data_matching_learned_settings"

    data_1, _ = common.load_data(common.DATASETS_DIR / "restaurant-1.csv")
    data_2, _ = common.load_data(common.DATASETS_DIR / "restaurant-2.csv")

    training_pairs = dedupe.training_data_link(data_1, data_2, "unique_id", 5000)
    true_dupes = get_true_dupes(data_1, data_2)

    return (data_1, data_2), settings_file, training_pairs, true_dupes


def get_true_dupes(data_1, data_2):
    all_data = data_1.copy()
    all_data.update(data_2)

    return common.get_true_dupes(all_data)


def run(data: tuple[dict, dict], settings_file, training_pairs, kwargs):
    data_1, data_2 = data

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

    print("clustering...")
    return deduper.join(data_1, data_2, **kwargs)


def make_report(true_dupes, clustering):
    print("Evaluate Clustering")
    predicted_dupes = set(frozenset(pair) for pair, _ in clustering)
    return common.Report.from_scores(true_dupes, predicted_dupes)


def cli():
    common.configure_logging()

    data, settings_file, training_pairs, true_dupes = load()

    for kwargs in [
        {"threshold": 0.5},
        {"threshold": 0.5, "constraint": "many-to-one"},
    ]:
        print()
        print(f"running with kwargs: {kwargs}")
        t0 = time.time()
        clustering = run(data, settings_file, training_pairs, kwargs=kwargs)
        elapsed = time.time() - t0

        print(make_report(true_dupes, clustering))
        print(f"ran in {elapsed} seconds")


if __name__ == "__main__":
    cli()
