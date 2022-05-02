import itertools
import os
import time

import dedupe

import common, canonical_matching


def load():
    settings_file = common.DATASETS_DIR / "canonical_gazetteer_learned_settings"

    data_1, _ = common.load_data(common.DATASETS_DIR / "restaurant-1.csv")
    data_2, _ = common.load_data(common.DATASETS_DIR / "restaurant-2.csv")

    training_pairs = dedupe.training_data_link(data_1, data_2, "unique_id", 5000)
    true_dupes = canonical_matching.get_true_dupes(data_1, data_2)

    return (data_1, data_2), settings_file, training_pairs, true_dupes


def run(data: tuple[dict, dict], settings_file, training_pairs):
    data_1, data_2 = data

    if os.path.exists(settings_file):
        with open(settings_file, "rb") as f:
            gazetteer = dedupe.StaticGazetteer(f)
    else:
        fields = [
            {"field": "name", "type": "String"},
            {"field": "address", "type": "String"},
            {"field": "cuisine", "type": "String"},
            {"field": "city", "type": "String"},
        ]

        gazetteer = dedupe.Gazetteer(fields)
        gazetteer.prepare_training(data_1, data_2, sample_size=10000)
        gazetteer.mark_pairs(training_pairs)
        gazetteer.train()

        with open(settings_file, "wb") as f:
            gazetteer.write_settings(f)

    gazetteer.index(data_2)
    gazetteer.unindex(data_2)
    gazetteer.index(data_2)

    print("clustering...")
    return gazetteer.search(data_1, n_matches=1, generator=True)


def make_report(true_dupes, clustering):
    print("Evaluate Clustering")
    predicted_dupes = set(
        frozenset([a, b]) for a, result in clustering for b, score in result
    )
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
